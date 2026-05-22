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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Le RAG facile !
 */
public class TestRoutage {

    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    // Méthode pour ingérer un document et retourner un EmbeddingStore
    private static EmbeddingStore<TextSegment> ingerer(String nomFichier,
                                                       EmbeddingModel embeddingModel) {
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(nomFichier, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        return embeddingStore;
    }

    // Méthode pour créer un ContentRetriever à partir d'un EmbeddingStore
    private static ContentRetriever creerRetriever(EmbeddingStore<TextSegment> embeddingStore,
                                                   EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    public static void main(String[] args) {

        configureLogger();

        String llmKey = System.getenv("GEMINI_KEY");
        if (llmKey == null || llmKey.isBlank()) {
            System.out.println("La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        // Mettre une température qui ne dépasse pas 0,3.
        // Le RAG sert à mieux contrôler l'exactitude des informations données par le LLM
        // et il est donc logique de diminuer la température.
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();;

        // -- Phase 1 du RAG : enregistrement des embeddings --

        // Création du modèle d'embedding
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // // -- PHASE 1 : Ingestion des 2 documents --

        // Calcul des embeddings
        EmbeddingStore<TextSegment> storeRag =
                ingerer("rag.pdf", embeddingModel);

        // Enregistrement dans le magasin d'embeddings
        EmbeddingStore<TextSegment> storeHadoop =
                ingerer("HadoopSparkMapReduce_1.pdf", embeddingModel);


        // -- PHASE 2 : Récupération et réponse --

        // 2 ContentRetrievers
        ContentRetriever retrieverRag = creerRetriever(storeRag, embeddingModel);
        ContentRetriever retrieverHadoop = creerRetriever(storeHadoop, embeddingModel);

        // Map avec descriptions pour le LM
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retrieverRag,
                "Cours sur le RAG (Retrieval-Augmented Generation), le fine-tuning et l'IA");
        descriptions.put(retrieverHadoop,
                "Cours sur Hadoop, Spark et MapReduce pour le traitement de données massives");

        // QueryRouter basé sur le LM
        QueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);

        // RetrievalAugmentor avec le QueryRouter
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Mémoire pour 10 messages
        MessageWindowChatMemory memory = MessageWindowChatMemory.withMaxMessages(10);


        // Création de l'assistant conversationnel, avec une mémoire.
        // L'implémentation de Assistant est faite par LangChain4j.
        // L'assistant gardera en mémoire les 10 derniers messages.
        // La base vectorielle en mémoire est utilisée pour retrouver les embeddings.
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Pour plusieurs questions sans recompiler
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


