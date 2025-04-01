from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import ServerlessSpec
import time  # To add a delay if needed

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

# ✅ Check if index already exists before creating
existing_indexes = [index["name"] for index in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # Your embedding model dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    time.sleep(2)  # Wait for the index to be ready

    # Agar index naya hai toh sirf tabhi data load karo
    print("Loading data from PDF...")
    extracted_data = load_pdf_file(data='Data/')
    text_chunks = text_split(extracted_data)
    embeddings = download_hugging_face_embeddings()

    # Insert data into Pinecone
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings,
    )
    print("Data inserted into Pinecone.")
else:
    print(f"Index '{index_name}' already exists. Skipping data loading.")

    # Agar index already exist karta hai toh sirf use load karo
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=download_hugging_face_embeddings(),
    )
    print("Existing data loaded from Pinecone.")



# # Extracting the Data
# extracted_data = load_pdf_file(data='Data/')
# text_chunks = text_split(extracted_data)
# embeddings = download_hugging_face_embeddings()




# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index_name = "medicalbot"




# # ✅ Check if index already exists before creating
# existing_indexes = [index["name"] for index in pc.list_indexes()]

# if index_name not in existing_indexes:
#     print(f"Creating new index: {index_name}")
#     pc.create_index(
#         name=index_name,
#         dimension=384,  # Your embedding model dimension
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
#     time.sleep(2)  # Wait for the index to be ready
# else:
#     print(f"Index '{index_name}' already exists. Skipping creation.")

    

# # Load Existing Index Instead of Recreating
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings,
# )

# pc = Pinecone(api_key=PINECONE_API_KEY)
# index_name = "medicalbot"

# pc.create_index(
#             name=index_name,
#             dimension=384,  # Your embedding model dimension
#             metric="cosine",
#             spec=ServerlessSpec(
#                 cloud="aws",
#                 region="us-east-1"
#                 )
#         )

# #  Load Existing Index Instead of Recreating**
# docsearch = PineconeVectorStore.from_documents(
#     documents = text_chunks,
#     index_name = index_name,
#     embedding = embeddings,
# )
