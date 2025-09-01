package com.example.opt_llm.advisors.expension.rag;

import lombok.Builder;
import lombok.Getter;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisor;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static com.example.opt_llm.advisors.expension.ExpansionQueryAdvisor.ENRICHED_QUESTION;

@Builder
public class RagAdvisor implements BaseAdvisor {

    private static final PromptTemplate template = PromptTemplate.builder().template("""
            CONTEXT: {context}
            Question: {question}
            """).build();

    private VectorStore vectorStore;

    @Builder.Default
//    private SearchRequest searchRequest = SearchRequest.builder().topK(1).similarityThreshold(0.62).build();
    private SearchRequest searchRequest = SearchRequest.builder().topK(4).similarityThreshold(0.62).build();

    @Getter
    private final int order;

    public static RagAdvisorBuilder build(VectorStore vectorStore) {
        return new RagAdvisorBuilder().vectorStore(vectorStore);
    }

    @Override
    public ChatClientRequest before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain) {

        String originalUserQuestion = chatClientRequest.prompt().getUserMessage().getText();
        String queryToRag = chatClientRequest.context().getOrDefault(ENRICHED_QUESTION, originalUserQuestion).toString();


        // searchRequest.getTopK()*2 - для того, чтобы ближайшие к границе изначального topK тоже захватить для переранжирования
        // после сортировки все отрежется до исходного topK (но в нем уже будет правильный ранк)

        List<Document> documents = vectorStore.similaritySearch(SearchRequest.from(searchRequest).query(queryToRag).topK(searchRequest.getTopK()*2).build());
        if (documents == null || documents.isEmpty()) {
            return chatClientRequest.mutate().context("CONTEXT", "ТУТ ПУСТО - ни один документ моя собачка не обнаружила").build();
        }

        BM25RerankEngine rerankEngine = BM25RerankEngine.builder().build();
        documents = rerankEngine.rerank(documents, queryToRag, searchRequest.getTopK());

        String llmContext = documents.stream().map(Document::getText).collect(Collectors.joining(System.lineSeparator()));


        String finalUserPromt = template.render(
                Map.of(
                        "context", llmContext,
                        "question", originalUserQuestion
                )
        );



        return chatClientRequest.mutate().prompt(chatClientRequest.prompt().augmentUserMessage(finalUserPromt)).build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain) {
        return chatClientResponse;
    }

}
