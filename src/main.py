from src.data_preprocessing.load_dataset import load_dataset
import dotenv


def main():
    dotenv.load_dotenv()
    seed = 0
    dataset = load_dataset(seed=seed)


if __name__ == "__main__":
    main()
