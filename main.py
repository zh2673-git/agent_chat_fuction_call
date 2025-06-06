from agent import ChatAgent

def main():
    agent = ChatAgent("openai")
    while True:
        user_input = input("用户输入: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Agent回复:", agent.run(user_input))

if __name__ == "__main__":
    main()
