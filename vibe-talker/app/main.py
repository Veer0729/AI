import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from .graph import create_chat_graph
MONGODB_URI = "mongodb://localhost:27017"
config = {"configurable": {"thread_id": "1"}}

def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer = checkpointer)

        # Initialize the recognizer
        r = sr.Recognizer()

        with sr.Microphone() as source:
            r.pause_threshold = 1

            while True:
                print("Say something!")
                audio = r.listen(source)

                print("processing: ")
                sst = r.recognize_google(audio)

                print("you said: ", sst)
                for event in graph.stream({"messages": [{"role": "user", "content": sst}]}, config, stream_mode="values"):
                    if "messages" in event:
                        event["messages"][-1].pretty_print()

main()

