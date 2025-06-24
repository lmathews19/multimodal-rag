#pip install -Uq "unstructured[all-docs]" pillow lxml pillow
#pip install -Uq chromadb tiktoken
#pip install -Uq langchain langchain-community langchain-openai langchain-groq
#pip install -Uq python_dotenv

import sys
import os
import warnings
# Suppress the max_size deprecation warning
warnings.filterwarnings("ignore", message=".*max_size.*deprecated.*")

from unstructured.partition.pdf import partition_pdf

# Check command line arguments
if len(sys.argv) != 2:
    print("Usage: python mm.py <pdf_file_path>")
    print("Example: python mm.py ./content/sample_w_table.pdf")
    sys.exit(1)

# Get file path from command line argument
file_path = sys.argv[1]

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
    print("Please provide a valid path to a PDF file.")
    sys.exit(1)

# Check if file is a PDF
if not file_path.lower().endswith('.pdf'):
    print(f"Error: File '{file_path}' is not a PDF file.")
    print("Please provide a path to a PDF file.")
    sys.exit(1)

print(f"Processing PDF file: {file_path}...")

# keys for the services we will use

output_path = "./content/"
# file_path = output_path + 'sample_w_table.pdf'  # Remove this line since we get it from command line

# Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True,            # extract tables
    strategy="hi_res",                     # mandatory to infer tables

    extract_image_block_types=[],           # No image types - "Image"
    # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

    extract_image_block_to_payload=False,   # Do not extract images as base64

    chunking_strategy="by_title",          # or 'basic'
    max_characters=1000,                  # defaults to 500
    combine_text_under_n_chars=100,       # defaults to 0
    new_after_n_chars=300,

    # extract_images_in_pdf=True,          # deprecated
)

# We get 2 types of elements from the partition_pdf function
set([str(type(el)) for el in chunks])

# separate tables from texts
tables = []
texts = []

for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)

    if "CompositeElement" in str(type((chunk))):
        texts.append(chunk)

print ("Done chunking.\n\n")
# Get the images from the CompositeElement objects
""" def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images = get_images_base64(chunks)
 """

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}

"""
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain - using OpenAI instead of Groq
model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

print ("Summarizing text portion of the document.\n\n")

# Summarize text
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

print ("Summarizing tables portion of the document.\n\n")

# Summarize tables
tables_html = [table.metadata.text_as_html for table in tables]
table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})


import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

print ("Adding text portion of the document to vector store.\n\n")

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

print ("Adding tables portion of the document to vector store.\n\n")

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

""" # Add image summaries
img_ids = [str(uuid.uuid4()) for _ in images]
summary_img = [
    Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
]
if summary_img:
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))

# Retrieve
docs = retriever.invoke(
    "Speed records"
)

print("\n\nSpeed records?\n\n")
print("Response: ")
for doc in docs:
    print(str(doc) + "\n\n" + "-" * 80)
 """

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    print("...")
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    # Return after all images are appended
    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


""" chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

response = chain.invoke(
    "Which are the 5 examples discussed in this document?"
)

print("\n\nHow many examples are included in this document?\n\n")
print(response)
 """

chain_with_sources = {
    "context": retriever | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(
        RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
)

"""
response = chain_with_sources.invoke(
    "Who has the land speed record?"
)

print("\n\nWho has the land speed record?\n\n")
print("Response:", response['response'])


print("\n\nContext:")
for text in response['context']['texts']:
    print(text.text)
    print("Page number: ", text.metadata.page_number)
    print("\n" + "-"*50 + "\n")

 for image in response['context']['images']:
    display_base64_image(image)
 """

# Interactive question-answering loop
print("\n" + "="*60)
print("Interactive Q&A Session")
print("Ask questions about the document. Type 'bye' to exit.")
print("="*60)

while True:
    # Get user input
    user_question = input("\nEnter your question: ").strip()
    
    # Check if user wants to exit
    if user_question.lower() == 'bye':
        print("Goodbye! Thanks for using the RAG system.")
        break
    
    # Skip empty questions
    if not user_question:
        print("Please enter a question.")
        continue
    
    try:
        # Get response using the chain with sources
        response = chain_with_sources.invoke(user_question)
        
        # Print the response
        print("\n" + "-"*40)
        print("Answer:")
        print(response['response'])
        print("-"*40)
        
        # Optionally show sources (uncomment if you want to see them)
        """
        print("\nSources:")
        for text in response['context']['texts']:
            print(f"Page {text.metadata.page_number}: {text.text[:200]}...")
        """
        
    except Exception as e:
        print(f"Error processing your question: {e}")
        print("Please try again with a different question.")