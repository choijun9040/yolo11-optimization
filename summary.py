import torch
from torchinfo import summary
from argparse import ArgumentParser

# Import the model definition from the nets/nn.py file
from nets import nn

def main():
    """
    Script to print the summary of the YOLOv11 model
    Allows specifying the model version and input size via the command line
    """
    parser = ArgumentParser(description="YOLOv11 Model Summary")
    parser.add_argument(
        '--model', 
        default='n', 
        type=str, 
        help="Model version: n, t, s, m, l, x"
    )
    parser.add_argument(
        '--input_size', 
        default=640, 
        type=int, 
        help="Image input size (e.g., 640, 640)"
    )
    parser.add_argument(
        '--num-classes',
        default=16,
        type=int,
        help="Number of classes for the model"
    )
    args = parser.parse_args()

    # Map the corresponding model creation function based on the model version
    model_factory = {
        'n': nn.yolo_v11_n
    }

    model_func = model_factory.get(args.model.lower())
    
    if not model_func:
        print(f"Error: Model version '{args.model}' is not a valid option.")
        print(f"Available options are: {list(model_factory.keys())}")
        return

    # Create model instance
    print(f"\nGenerating summary for yolo_v11_{args.model} with input size {args.input_size}x{args.input_size}...")
    model = model_func(num_classes=args.num_classes)
    
    # Define input shape
    input_shape = (1, 3, args.input_size, args.input_size)

    # Print summary using torchinfo
    summary(
        model, 
        input_size=input_shape,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=5
    )

if __name__ == "__main__":
    main()