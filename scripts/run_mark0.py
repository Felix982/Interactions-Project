from src.config import Mark0Config
from src.models.mark0 import Mark0Model


def main() -> None:
    config = Mark0Config()
    model = Mark0Model(config)

    history = model.run(2500)

    print("Mark 0 model initialized successfully.")
    print("Initial unemployment:", history["unemployment"][0])
    print("Final unemployment:", history["unemployment"][-1])
    print("Initial average price:", history["avg_price"][0])
    print("Final average price:", history["avg_price"][-1])
    print("First five firm demands:", model.d[:5])
    print("Initial active firms:", history["active_firms"][0])
    print("Final active firms:", history["active_firms"][-1])

if __name__ == "__main__":
    main()