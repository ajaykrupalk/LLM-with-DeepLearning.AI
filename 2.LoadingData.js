import dotenv from "dotenv";
import { GithubRepoLoader } from "langchain/document_loaders/web/github";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import ignore from "ignore";
import * as parse from "pdf-parse";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CharacterTextSplitter } from "langchain/text_splitter";

dotenv.config();

let res;

// Using Document Loaders as source for the LLM
// In this case, we are using GitHub
// Will not include anything under ignorePaths
// const loader = new GithubRepoLoader(
//     "https://github.com/langchain-ai/langchainjs",
//     {
//         recursive: false, //Working with packages
//         ignorePaths: ["*.md", "yarn.lock"]
//     }
// )
// const docs = await loader.load();
// console.log(docs.slice(0,3));

// Loading Data from a PDF file
const loader = new PDFLoader("./files/drylab.pdf");
const docs = await loader.load();
// console.log(docs.slice(0,3));

// Splitting Data 
// Splits the text by different characters until it finds a suitable character
// const splitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
//     chunkSize: 32,
//     chunkOverlap: 0,
// });
// const code = `function helloWorld(){
//     console.log("Hello World!");
// }
// console.log("Hello World!");
// helloWorld();`
// res = await splitter.splitText(code);

// Splits the text by a specificied separator
// const splitter = new CharacterTextSplitter({
//     chunkSize: 32,
//     chunkOverlap: 0,
//     separator: " "
//   });

// res =await splitter.splitText(code);

// const splitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
//     chunkSize: 64,
//     chunkOverlap: 32,
// });

// res = await splitter.splitText(code);

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 512,
    chunkOverlap: 64,
});
const splitDocs = await splitter.splitDocuments(docs);
console.log(splitDocs.slice(0, 5));