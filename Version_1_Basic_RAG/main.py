from chatbot import respond
from rag import search

while True:
    user_input=input("You😉: ").lower()
    if "exit" in user_input or "bye" in user_input:
        print("Jarvis🤖: See Yaa")
        break
    print("📑 Context: ",end="")
    print(search(user_input, top=5))
    print()
    print("Jarvis🤖: ",end="")
    print(respond(user_input))
    print()