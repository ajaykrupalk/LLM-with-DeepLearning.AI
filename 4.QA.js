// Peer dependency
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import * as parse from "pdf-parse";

let res;

export async function loadAndSplitChunks({
    chunkSize,
    chunkOverlap
}) {
    const loader = new PDFLoader("./files/drylab.pdf");

    const rawCS229Docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap,
    });

    const splitDocs = await splitter.splitDocuments(rawCS229Docs);
    return splitDocs;
}

export async function initializeVectorstoreWithDocuments({
    documents
}) {
    const embeddings = new GoogleGenerativeAIEmbeddings({
        modelName: "embedding-001", // 768 dimensions
        taskType: TaskType.RETRIEVAL_DOCUMENT,
        title: "Retrieval Document",
    });;
    const vectorstore = new MemoryVectorStore(embeddings);
    await vectorstore.addDocuments(documents);
    return vectorstore;
}

export function createDocumentRetrievalChain() {
    const convertDocsToString = (documents) => {
        return documents.map((document) => `<doc>\n${document.pageContent}\n</doc>`).join("\n");
    };

    const documentRetrievalChain = RunnableSequence.from([
        (input) => input.standalone_question,
        retriever,
        convertDocsToString,
    ]);

    return documentRetrievalChain;
}

export function createRephraseQuestionChain() {
    const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.`;

    const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
        ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
        new MessagesPlaceholder("history"),
        ["human", "Rephrase the following question as a standalone question:\n{question}"],
    ]);
    const rephraseQuestionChain = RunnableSequence.from([
        rephraseQuestionChainPrompt,
        new ChatGoogleGenerativeAI({
            modelName: "gemini-pro",
            maxOutputTokens: 2048,
            apiKey: process.env.GOOGLE_API_KEY
        }),
        new StringOutputParser(),
    ]);
    return rephraseQuestionChain;
}

import dotenv from "dotenv";
dotenv.config();

const splitDocs = await loadAndSplitChunks({
    chunkSize: 1536,
    chunkOverlap: 128
})

const vectorStore = await initializeVectorstoreWithDocuments({
    documents: splitDocs
})
const retriever = vectorStore.asRetriever();

// Converting the retrieved documents to a string
const convertDocsToString = (documents) => {
    return documents.map((document) => {
        return `<doc>\n${document.pageContent}</doc>`
    }).join("\n")
};

const documentRetrievalChain = RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString
])
const results = await documentRetrievalChain.invoke({
    question: "What are the prerequisites for this course?"
});
// console.log(results);

const TEMPLATE_STRING = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the provided context, answer the user's question 
to the best of your ability using only the resources provided. 
Be verbose!

<context>

{context}

</context>

Now, answer this question using the above context:

{question}`;

const answerGenerationPrompt = ChatPromptTemplate.fromTemplate(
    TEMPLATE_STRING
);

import { RunnableMap } from "@langchain/core/runnables"

// map a list of inputs to a list of outputs by invoking each input with the corresponding runnable.
// const runnableMap = RunnableMap.from({
//     context: documentRetrievalChain,
//     question: (input) => input.question
// });

// res = await runnableMap.invoke({
//     question: "What are the prerequisites for this course?"
// })

// console.log(res)

const model = new ChatGoogleGenerativeAI({
    modelName: "gemini-pro",
    maxOutputTokens: 2048,
    apiKey: process.env.GOOGLE_API_KEY
})

const retrievalChain = RunnableSequence.from([
    {
        context: documentRetrievalChain,
        question: (input) => input.question,
    },
    answerGenerationPrompt,
    model,
    new StringOutputParser(),
]);
// res = await retrievalChain.invoke({
//     question: "What are the prerequisites for this course?"
// })
// console.log(res)

// const followupAnswer = await retrievalChain.invoke({
//     question: "Can you list them in bullet point form?"
// });
// console.log(followupAnswer);

const docs = await documentRetrievalChain.invoke({
    question: "Can you list them in bullet point form?"
  });
  
console.log(docs);