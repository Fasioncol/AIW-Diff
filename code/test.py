import argparse

from validate import run_evaluation

def test():
    parser = argparse.ArgumentParser(description="Model test on CT-CTA dataset")
    parser.add_argument("--config", type=str, default="ct_cta.yml", help="Path to the config file.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the model checkpoint file (e.g., ckpt_5000.pth).")
    parser.add_argument("--data_dir", type=str, default="./data/test", help="Path to the test data directory.")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                        help="Directory to save test results.")
    parser.add_argument("--timesteps", type=int, default=100, help="Number of steps for sampling.")
    parser.add_argument("--eta", type=float, default=0.0, help="Eta for DDIM sampling (0.0 for deterministic).")
    args = parser.parse_args()
    run_evaluation(
        args.config, args.ckpt, args.data_dir, args.output_dir,
        args.timesteps, args.eta, "test",
    )

if __name__ == "__main__":
    test()
