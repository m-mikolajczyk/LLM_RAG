import utils

def main():
    print("Welcome to the Movie Search Assistant! \n ")
    print("Ask me anything about movies (e.g., 'Reccomend me movies similar to The Hunted')")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not query:
            print("Please enter a question.")
            continue

        response = utils.search_and_answer(query)
        print("\nAnswer:\n", response, "\n")

if __name__ == "__main__":
    main()