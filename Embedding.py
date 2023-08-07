from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import PyPDFLoader
import time
import os
import torch.nn as nn
import torch
from dotenv import load_dotenv



class Storing_docs(nn.Module) : 
    
    def __init__(self,vectorstore_path:str ,model_id:str, chunk_size:int, chunk_overlap:int) -> None:
        super().__init__()

        load_dotenv()

        self.vectorstore_path = vectorstore_path
        self.model_id = model_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap , separators=["\n", "\n\n", " "])
        self.embedding_model = HuggingFaceEmbeddings(model_name = self.model_id, model_kwargs = {'device' : self.device})
        self.vector_store = DeepLake(dataset_path=self.vectorstore_path, embedding_function=self.embedding_model)

    def Add_Doc_Dir(self, doc_dir : str) -> None:
        """
        from the documents of the given directory this method extracts text, chunks it, adds it and its embeddings to the vector store 
        """
        for item in os.listdir(doc_dir) : 
            doc_path = os.path.join(doc_dir, item)
            print(f"extracting document : {doc_path}")
            loader = PyPDFLoader(doc_path)
            text = loader.load()
            chunked_text = self.splitter.split_documents(text)
            self.vector_store.add_documents(
                chunked_text
            )
            print(f"{doc_path} successfully added !!")
        return 
    
    def Add_doc(self, doc_path : str) -> None : 
        """
        extract text from a document, chunk it, embed it and add it to the vector store
        """
        loader = PyPDFLoader(doc_path)
        text = loader.load()
        chunked_text = self.splitter.split_documents(text)
        self.vector_store.add_documents(
            chunked_text
        )
        print(f"{doc_path} successfully added !!")
        

    def search_similarity_by_query(self, query : str) -> list[str] :
        """
        this method retrieves the most similar document in the vectorstore with respect to a given query
        """
        t = time.perf_counter()
        retrieved_docs = self.vector_store.search(query=query, 
                                                  search_type='similarity')
        print(f"Retrieval operation took : {time.perf_counter() - t} s")

        return retrieved_docs
    
    def search_asimilarity_by_query(self, query : str) -> list[str] :
        """
        this method retrieves the most similar document in the vectorstore with respect to a given query
        """
        t = time.perf_counter()
        retrieved_docs = self.vector_store.asearch(query=query, 
                                                  search_type='similarity')
        print(f"Retrieval operation took : {time.perf_counter() - t} s")

        return retrieved_docs

    def delete(self, delete_vector_store : bool = False, empty_vector_store : bool = False, id_list: list[str] = None) -> None : 
        """
        calling this method can empty the vector store , delete all of it or remove an element of it 
        """
        if empty_vector_store :
            self.vector_store.delete(delete_all=empty_vector_store)
            print('vectore store is now empty')
        elif delete_vector_store :
            self.vector_store.force_delete_by_path(path=self.vectorstore_path)
            print('vectore store succefully deleted')
        elif id_list!=None:
            self.vector_store.delete(id_list)
            print('the document has succefully be deleted')


if __name__ == '__main__':
    load_dotenv() 
    doc_path="your_PDF.pdf"
    doc_dir="pdfs_dir"
    query=input('Write your request here:')
#***Before doing anything you need to create your activeloop account and select create a vector store DB
#**** To do so access to : https://app.activeloop.ai/ and create your account
    ins = Storing_docs(model_id="intfloat/e5-small-v2",
                vectorstore_path= "hub://your_activeloopaccount",
                 chunk_size=1000, chunk_overlap=0)
    ins.Add_doc(doc_path=doc_path)
    ins.Add_Doc_Dir(doc_dir=doc_dir)
    res = ins.search_similarity_by_query(query)
    print(res)
    #*****************Delete function*************************
    #ins.delete(empty_vector_store=True)
    #ins.delete(delete_vector_store=True)
