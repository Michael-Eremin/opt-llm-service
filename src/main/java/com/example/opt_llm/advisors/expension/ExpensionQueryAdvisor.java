package com.example.opt_llm.advisors.expension;

import lombok.Builder;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.ollama.api.OllamaOptions;

import java.util.Map;

@Builder
public class ExpensionQueryAdvisor implements BaseAdvisor {

    private static final PromptTemplate template = PromptTemplate.builder()
            .template("""
                    Если запрос содержит "из алгебры", переформулируй запрос так, чтобы вместо этого словосочетания было "из геометрии".
                    Если этого словосочетания в запросе нет, то верни запрос неизмененным.
                    
                    Question: {question}
                    Reformulated:
                    """).build();
    public static final String ENRICHED_QUESTION = "ENRICHED_QUESTION";
    public static final String ORIGINAL_QUESTION = "ORIGINAL_QUESTION";
    public static final String EXPENSION_RATIO = "EXPENSION_RATIO";

    private ChatClient chatClient;

    private final ChatModel chatModel;

    public static ExpensionQueryAdvisorBuilder builder(ChatModel chatModel1) {
        return new ExpensionQueryAdvisorBuilder().chatClient(ChatClient.builder(chatModel1)
                        .defaultOptions(OllamaOptions.builder()
                                .temperature(0.0)
                                .topK(1)
                                .topP(0.1)
                                .repeatPenalty(1.0)
                                .build()
                        )
                .build());
    }

    private int order;

    @Override
    public ChatClientRequest before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain) {
        String userQuestion =  chatClientRequest.prompt().getUserMessage().getText();
        String enrichedQuestion = chatClient
                .prompt()
                .user((template.render(Map.of("question", userQuestion))))
                .call()
                .content();

        double ratio = enrichedQuestion.length() / (double) userQuestion.length();

        return chatClientRequest.mutate()
                .context(ENRICHED_QUESTION, enrichedQuestion)
                .context(ORIGINAL_QUESTION, userQuestion)
                .context(EXPENSION_RATIO, ratio)

                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain) {
        return chatClientResponse;
    }

    @Override
    public int getOrder() {
        return 0;
    }
}
