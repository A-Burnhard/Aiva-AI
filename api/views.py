from django.shortcuts import render
from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import DataSerializer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .models import Chat
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

#Langchainfrom langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch, Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, ChatMessage
from langchain.tools import format_tool_to_openai_function, YouTubeSearchTool, MoveFileTool, BaseTool, StructuredTool, tool
from langchain.agents import ConversationalAgent, Tool, initialize_agent
from langchain.pydantic_v1 import BaseModel, Field
import langchain
import openai
import json
import datetime
import os
import requests


#SETTING UP QA CHAIN
openai.api_key = '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'
openai_api_key = openai.api_key
llm = ChatOpenAI(model="gpt-4-0613", openai_api_key=openai.api_key)
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
#file_path= loader = "/home/bernard/SR/static/base/assets/js/data.pdf"
file_path= "data.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pages)
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = DocArrayInMemorySearch.from_documents(documents, embedding)
k=10
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

#- Prompt Template
context= "receptionist"
template = """As a receptionist of Nerasol, I'm ready to answer your questions about the company or its services. Please ask me a specific question.{context}.Question: {question} """

#- Define the QA model
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", openai_api_key=openai_api_key, temperature=1)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)


# CUSTOM FUNCTIONS AND TOOLS

@tool
def send_sms(action_input):
    """
    Sends an SMS to the specified phone number with the given message.

    Parameters:
    - number (str): The phone number of the recipient (e.g., "0550916600").
    - message (str): The content of the SMS.

    Returns:
    - dict: A dictionary containing the result of the SMS sending operation.
    """
    number, message = action_input.split(",")
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


#CREATING A CONVERSATIONAL AGENT
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

from langchain.agents import initialize_agent


tools = [

    Tool(
        name='Nerasol Knowledge Base',
        func=qa_chain.run,
        description=(
            'use this tool when answering questions as a receptionist of Nerasol '

        )
    ),
    Tool(
        name='Send SMS',
        func=send_sms,  # Assuming your `send_sms` function is defined
        description='Use this tool to send SMS, providing action input in a "phone number, message" format'
        #handle_parsing_errors=True
    ),
    

]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

fixed_prompt = '''As a receptionist assistant, I am designed to efficiently manage various administrative tasks and provide assistance to visitors and employees.
I can handle inquiries about company policies. Extract phone numbers from Nerasol knowledge base and use that phone number to  SMS to the person in question , and provide general information about the organization's services and facilities. My goal is to streamline communication and ensure a positive experience for everyone who interacts with the company.
While I excel at handling routine inquiries and administrative tasks, please note that I may not be equipped to answer questions outside the scope of receptionist duties. For inquiries related to complex technical issues, company-specific protocols, or sensitive matters, it's advisable to consult with the appropriate department or personnel.
My capabilities are continually evolving, and I strive to enhance efficiency and effectiveness in providing assistance to both guests and staff members. Whether it's directing visitors to their appointments or assisting employees with administrative tasks, I am here to support the smooth operation of the reception area and contribute to a welcoming and professional environment.
Please feel free to reach out to me for assistance with reception-related inquiries and tasks. I am committed to delivering prompt and courteous service to meet the needs of our guests and employees. If i dont know anything i will say i dont know.
'''

agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt




#@method_decorator(csrf_exempt, name='dispatch')
class ChatView(APIView):
    def post(self, request):
        try:
            print("Request Data:", request.data)
            serializer = DataSerializer(data=request.data)
            if serializer.is_valid():
                user_message = serializer.validated_data.get('user_message', '')

                # Generate bot response
                bot_response = agent(user_message)['output']

                # Save both user message and bot response
                data_instance = Chat.objects.create(user_message=user_message)

                # Prepare response data including only the 'output' part of the bot response
                response_data = {
                    'output': bot_response
                }

                return Response(response_data, status=status.HTTP_201_CREATED)

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
        

