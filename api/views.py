from django.shortcuts import render
from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import DataSerializer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from base.models import Chat
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

#Langchain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import format_tool_to_openai_function, tool
#from langchain.globals import set_llm_cache, get_llm_cache 
#New
#from langchain import HumanMessage, AIMessage, ChatMessage
from langchain_community.tools import format_tool_to_openai_function, YouTubeSearchTool, MoveFileTool
from langchain_community.vectorstores import Chroma
import openai
import datetime
import os
import requests 

# Initialize the OpenAI API client with your API key

openai.api_key = 
openai_api_key = openai.api_key
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-4-0613"
else:
    llm_model = "gpt-4-0613"

#Internal Data Implementation
#- Load the data, split it into chunks, and embed it
loader = PyPDFLoader("data.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pages)
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = DocArrayInMemorySearch.from_documents(documents, embedding)
k=10
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

#- Prompt Template
context= "receptionist"
template = """As a receptionist of Tech company called Nerasol.Use this context,reply to messages in three sentences or less :{context}.Question: {question} """

#- Define the QA model
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", openai_api_key=openai_api_key, temperature=0.9)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)


#Function DEScription
function_description = [{
    "name": "send_sms",
    "description": "Send an SMS to a specified phone number with a given message.",
    "parameters": {
        "type": "object",
        "properties": {
            "number": {
                "type": "string",
                "description": "The phone number of the recipient (e.g., '0550916600').",
            },
            "message": {
                "type": "string",
                "description": "The content of the SMS message.",
            },
        },
        "required": ["number", "message"],
    },
}
]

#TOOLS
 # Assuming you have the requests library installed

@tool
def send_sms(number, message):
    """
    Sends an SMS to the specified phone number with the given message.

    Parameters:
    - number (str): The phone number of the recipient (e.g., "0550916600").
    - message (str): The content of the SMS.

    Returns:
    - dict: A dictionary containing the result of the SMS sending operation.
    """
    # Construct the payload for the POST request
    payload = {
        "number": number,
        "message": message,
    }

    # Make a POST request to the specified URL
    url = "https://laws.adudor.com/api/fire-sms"
    response = requests.post(url, json=payload)

    # Return the result as a dictionary
    result = {
        "status_code": response.status_code,
        "response_content": response.json() if response.headers['content-type'] == 'application/json' else response.text,
    }

    return result


#Function calling Implementation
#Creating Tools

send_sms_tool = format_tool_to_openai_function(send_sms)
#qa_chain.register_tool(send_sms_tool)








@method_decorator(csrf_exempt, name='dispatch')
class ChatView(APIView):
    def post(self, request):
        try:
            serializer = DataSerializer(data=request.data)
            if serializer.is_valid():
                user_message = serializer.validated_data.get('user_message', '')
                response = qa_chain(user_message)
                bot_response = response['result']

                data_instance = serializer.save(bot_response=bot_response)

                return Response(DataSerializer(data_instance).data, status=status.HTTP_201_CREATED)

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            # Log the exception for further analysis
            print("Exception:", str(e))
            return Response({'error': 'Internal Server Error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class TestView(generics.CreateAPIView):
    queryset = Chat.objects.all()
    serializer_class = DataSerializer

    def perform_create(self, serializer):
        user_message = serializer.validated_data['user_message']
        response = qa_chain(user_message)
        response = response['result']
        bot_response = response

        serializer.save(bot_response=bot_response)


# Chat Model
chat = ChatOpenAI(temperature=0.4, model=llm_model, openai_api_key=openai_api_key)
class ProcessorView(APIView):
    def post(self, request):
        try:
            serializer = DataSerializer(data=request.data)
            if serializer.is_valid():
                # Get the text from the request
                input_text = serializer.validated_data['input_text']

                # Create a prompt for OpenAI using the input text
                prompt = f"Please provide assistance as a receptionist of a tech company called Nerasol: \n```{input_text}```"
                prompt_template = ChatPromptTemplate.from_template(prompt)
                reception_messages = prompt_template.format_messages(
                    style="Receptionist",
                    text=input_text,
                    max_tokens=500,
                )

                # Use OpenAI to generate a response
                response = chat(reception_messages)

                # Extract the generated response from OpenAI
                processed_text = response

                # Create a new instance and save it with the processed text
                data_instance = serializer.create({
                    'processed_text': processed_text
                })

                # Return the serialized data to the frontend
                return Response(DataSerializer(data_instance).data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            # Handle exceptions here, you can log the exception for debugging
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        