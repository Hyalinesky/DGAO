import argparse
from datasets import Dataset
from trl import DGAOTrainer, DGAOConfig
from transformers import AutoTokenizer
import json
import torch

def load_order_fair_data(file_path, tokenizer, num_orders=8, max_prompt_tokens=2048, max_total_tokens=3072):
    """
    Load data from jsonl file and format for order-fair training.
    
    Args:
        file_path: Path to the jsonl file
        tokenizer: Tokenizer for counting tokens
        num_orders: Number of orders per group (should be 8)
        max_prompt_tokens: Maximum tokens allowed for a single prompt (default: 2048)
        max_total_tokens: Maximum total tokens for the conversation (default: 3072)
    
    Returns:
        List of formatted data with prompt_variants and answers
    """
    data_items = []
    
    # Read jsonl file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data_items.append(json.loads(line))
    
    print(f"Loaded {len(data_items)} items from {file_path}")
    
    # Check if the number of items is divisible by num_orders
    if len(data_items) % num_orders != 0:
        print(f"Warning: Total items ({len(data_items)}) is not divisible by num_orders ({num_orders})")
        # Trim to make it divisible
        data_items = data_items[:len(data_items) // num_orders * num_orders]
        print(f"Trimmed to {len(data_items)} items")
    
    formatted_data = []
    skipped_groups = 0
    
    # Process data in groups of num_orders
    for i in range(0, len(data_items), num_orders):
        group = data_items[i:i + num_orders]
        
        # Check token lengths for all items in the group
        skip_group = False
        for item in group:
            # Create messages for token counting
            messages = [
                {"role": "system", "content": item.get("system", "You are a helpful assistant.")},
                {"role": "user", "content": item["prompt"]}
            ]
            
            # Count tokens for the prompt only (user message)
            prompt_tokens = len(tokenizer.encode(item["prompt"], add_special_tokens=False))
            
            # Count tokens for the entire conversation
            conversation_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            total_tokens = len(tokenizer.encode(conversation_text, add_special_tokens=True))
            
            # Check if prompt exceeds max_prompt_tokens or total exceeds max_total_tokens
            if prompt_tokens > max_prompt_tokens:
                print(f"Skipping group {i//num_orders + 1}: Prompt too long ({prompt_tokens} > {max_prompt_tokens} tokens)")
                skip_group = True
                break
            elif total_tokens > max_total_tokens:
                print(f"Skipping group {i//num_orders + 1}: Total conversation too long ({total_tokens} > {max_total_tokens} tokens)")
                skip_group = True
                break
        
        if skip_group:
            skipped_groups += 1
            continue
        
        # Check if all responses in the group are consistent
        responses = [item["response"] for item in group]
        if not all(resp == responses[0] for resp in responses):
            print(f"Warning: Inconsistent responses in group starting at index {i}")
            print(f"Responses: {responses}")
        
        # Extract the common answer (response)
        answer = responses[0]
        
        # Construct prompt variants for this group
        prompt_variants = []
        for item in group:
            # Format as conversational messages
            messages = [
                {"role": "system", "content": item.get("system", "You are a helpful assistant.")},
                {"role": "user", "content": item["prompt"]}
            ]
            prompt_variants.append(messages)
        
        # Add to formatted data
        formatted_data.append({
            "prompt_variants": prompt_variants,
            "answer": answer
        })
    
    print(f"Created {len(formatted_data)} groups with {num_orders} variants each")
    print(f"Skipped {skipped_groups} groups due to token length constraints")
    print(f"Total groups processed: {len(data_items) // num_orders}")
    print(f"Success rate: {len(formatted_data)}/{len(data_items) // num_orders} ({len(formatted_data)/(len(data_items) // num_orders)*100:.1f}%)")
    
    return formatted_data

def validate_data_consistency(data_list, tokenizer, num_orders=8):
    """
    Validate that each group has the expected number of variants
    and print some statistics
    """
    print("\n=== Data Validation ===")
    print(f"Total groups: {len(data_list)}")
    print(f"Expected variants per group: {num_orders}")
    
    variant_counts = [len(item["prompt_variants"]) for item in data_list]
    print(f"Actual variants per group: {set(variant_counts)}")
    
    if len(set(variant_counts)) == 1 and variant_counts[0] == num_orders:
        print("✓ All groups have the correct number of variants")
    else:
        print("✗ Some groups have incorrect number of variants")
    
    # Show a sample with token counts
    if data_list:
        print("\n=== Sample Data with Token Counts ===")
        sample = data_list[0]
        print(f"Answer: {sample['answer']}")
        print(f"Number of prompt variants: {len(sample['prompt_variants'])}")
        
        # Check token counts for the first variant
        first_variant = sample['prompt_variants'][0]
        prompt_text = first_variant[1]["content"]  # User message
        prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        
        conversation_text = tokenizer.apply_chat_template(
            first_variant, 
            tokenize=False, 
            add_generation_prompt=True
        )
        total_tokens = len(tokenizer.encode(conversation_text, add_special_tokens=True))
        
        print(f"Sample prompt tokens: {prompt_tokens}")
        print(f"Sample total conversation tokens: {total_tokens}")
        
        print("First variant:")
        for msg in sample['prompt_variants'][0]:
            print(f"  {msg['role']}: {msg['content'][:100]}...")
        print("Second variant:")
        for msg in sample['prompt_variants'][1]:
            print(f"  {msg['role']}: {msg['content'][:100]}...")

def train_dgao_model(
    model_name,
    file_path,
    output_dir,
    task,
    num_orders=8,
    max_prompt_tokens=468,
    max_total_tokens=596,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3.0,
    beta=0.05,
    save_steps=16,
    temperature=1.0,
    top_p=1.0,
    alpha=0.5,
    max_completion_length=128,
    max_prompt_length=468,
    steps_per_generation=1,
    num_iterations=1,
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=False
):
    """
    Train a DGAO model with the given parameters.
    
    Args:
        model_name: Path to the model to train
        file_path: Path to the training data file
        output_dir: Directory to save the trained model
        task: Task name
        num_orders: Number of orders per group
        max_prompt_tokens: Maximum tokens for prompt
        max_total_tokens: Maximum total tokens for conversation
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        num_train_epochs: Number of training epochs
        beta: Beta parameter for DGAO
        save_steps: Steps between saves
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        alpha: Alpha parameter for advantage mixing
        max_completion_length: Maximum completion length
        max_prompt_length: Maximum prompt length
        steps_per_generation: Steps per generation
        num_iterations: Number of iterations
        logging_steps: Steps between logging
        bf16: Whether to use bf16
        gradient_checkpointing: Whether to use gradient checkpointing
    """
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and process data
    print("Loading and processing data...")
    data_list = load_order_fair_data(
        file_path, 
        tokenizer,
        num_orders=num_orders,
        max_prompt_tokens=max_prompt_tokens,
        max_total_tokens=max_total_tokens
    )
    
    # Validate the data
    validate_data_consistency(data_list, tokenizer, num_orders)
    
    # Create a Hugging Face dataset
    dataset = Dataset.from_list(data_list)
    
    # Training configuration
    training_args = DGAOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=bf16,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        beta=beta,
        num_orders=num_orders,
        save_steps=save_steps,
        temperature=temperature,
        top_p=top_p,
        gradient_checkpointing=gradient_checkpointing,
        steps_per_generation=steps_per_generation,
        num_iterations=num_iterations,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
    )
    
    # Create trainer with alpha parameter for advantage mixing
    trainer = DGAOTrainer(
        model=model_name,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        alpha=alpha,
        task=task,
    )
    
    print("Starting training...")
    trainer.train()
    print("Training completed!")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DGAO model")
    
    # Required arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Path to the model to train")
    parser.add_argument("--file_path", type=str, required=True,
                       help="Path to the training data file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save the trained model")
    parser.add_argument("--task", type=str, required=True,
                       help="Task name")
    
    # Optional arguments with defaults
    parser.add_argument("--num_orders", type=int, default=8,
                       help="Number of orders per group (default: 8)")
    parser.add_argument("--max_prompt_tokens", type=int, default=468,
                       help="Maximum tokens for prompt (default: 468)")
    parser.add_argument("--max_total_tokens", type=int, default=596,
                       help="Maximum total tokens for conversation (default: 596)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                       help="Batch size per device (default: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                       help="Number of training epochs (default: 3.0)")
    parser.add_argument("--beta", type=float, default=0.05,
                       help="Beta parameter for DGAO (default: 0.05)")
    parser.add_argument("--save_steps", type=int, default=16,
                       help="Steps between saves (default: 16)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling parameter (default: 1.0)")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Alpha parameter for advantage mixing (default: 0.5)")
    parser.add_argument("--max_completion_length", type=int, default=128,
                       help="Maximum completion length (default: 128)")
    parser.add_argument("--max_prompt_length", type=int, default=468,
                       help="Maximum prompt length (default: 468)")
    parser.add_argument("--steps_per_generation", type=int, default=1,
                       help="Steps per generation (default: 1)")
    parser.add_argument("--num_iterations", type=int, default=1,
                       help="Number of iterations (default: 1)")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Steps between logging (default: 10)")
    parser.add_argument("--no_bf16", action="store_true",
                       help="Disable bf16 training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    
    return parser.parse_args()

def main():
    """Main function to run the training."""
    args = parse_arguments()
    
    # Convert args to kwargs for the training function
    training_kwargs = vars(args).copy()
    training_kwargs['bf16'] = not args.no_bf16  # Invert no_bf16 flag
    del training_kwargs['no_bf16']  # Remove the no_bf16 key
    
    # Call the training function
    train_dgao_model(**training_kwargs)

if __name__ == "__main__":
    main()