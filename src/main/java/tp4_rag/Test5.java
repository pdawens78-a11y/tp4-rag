package tp4_rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test5 {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {

        configureLogger();

        String llmKey = System.getenv("GEMINI_KEY");
        if (llmKey == null || llmKey.isBlank()) {
            System.out.println("La variable d'environnement GEMINI_2 n'est pas définie.");
            return;
        }

        String tavilyKey = System.getenv("TAVILY_API_KEY");
        if (tavilyKey == null || tavilyKey.isBlank()) {
            System.out.println("La variable d'environnement tavily_key n'est pas définie.");
            return;
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // ---- PHASE 1 : Ingestion du document RAG ----

        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ---- PHASE 2 : Récupération et réponse ----

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // WebSearchEngine Tavily
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        // ContentRetriever pour le Web
        ContentRetriever retrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();

        // DefaultQueryRouter avec les 2 ContentRetrievers
         /*
        QueryRouter queryRouter = new DefaultQueryRouter(contentRetriever, retrieverWeb);
        */

        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(contentRetriever,
                "Cours sur le RAG (Retrieval-Augmented Generation), le fine-tuning et l'IA");
        descriptions.put(retrieverWeb,
                "Informations récentes sur le Web, événements actuels, actualités");

        QueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);


        // RetrievalAugmentor avec le QueryRouter
        RetrievalAugmentor monRetrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Mémoire pour 10 messages
        MessageWindowChatMemory memory = MessageWindowChatMemory.withMaxMessages(10);


        // Création de l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .retrievalAugmentor(monRetrievalAugmentor)
                .build();

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("=============================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("=============================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("=============================================");
            }
        }
    }
}