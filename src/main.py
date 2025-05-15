import dotenv

from src.data_preprocessing.load_dataset import load_dataset
from src.runners.cb import train_content_based
from src.runners.hr import train_highest_rated
from src.runners.item_item import train_item_item
from src.runners.mf_als import train_mf_als
from src.runners.mf_als_mr import train_mf_als_mr
from src.runners.mf_sgd import train_mf_sgd
from src.runners.mp import train_most_popular
from src.runners.user_user import train_user_user


def main():
    dotenv.load_dotenv()
    seed = 0
    dataset = load_dataset()

    # train_mf_sgd(dataset, seed=seed)
    # train_mf_als(dataset, seed=seed)
    # train_mf_als_mr(dataset, seed=seed)
    #
    # train_user_user(dataset, seed=seed)
    # train_item_item(dataset, seed=seed)

    train_most_popular(dataset=dataset)
    train_highest_rated(dataset=dataset)

    # train_content_based(dataset, seed=seed)


if __name__ == "__main__":
    main()
