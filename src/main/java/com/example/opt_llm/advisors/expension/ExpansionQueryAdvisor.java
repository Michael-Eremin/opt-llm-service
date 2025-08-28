package com.example.opt_llm.advisors.expension;

import lombok.Builder;
import lombok.Getter;
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
public class ExpansionQueryAdvisor implements BaseAdvisor {



    private static final PromptTemplate template = PromptTemplate.builder()
            .template("""
                    Instruction: Расширь поисковый запрос, добавив наиболее релевантные термины.
                    
                    СПЕЦИАЛИЗАЦИЯ ПО МАТЕМАТИКЕ:
                    - Алгебраические термины: многочлен, факторизация, квадратный трёхчлен, тождество, разложение на множители
                    - Геометрические термины: площадь квадрата, прямоугольник, теорема Пифагора, катет и гипотенуза, геометрическая интерпретация формул
                    
                    ПРАВИЛА:
                    1. Сохрани ВСЕ слова из исходного вопроса
                    2. Добавь МАКСИМУМ ПЯТЬ наиболее важных терминов
                    3. Выбирай самые специфичные и релевантные слова
                    4. Результат - простой список слов через пробел
                    
                    СТРАТЕГИИ ВЫБОРА:
                    - Приоритет: специализированные термины
                    - Избегай общих слов
                    - Фокусируйся на ключевых понятиях
                    
                    ПРИМЕРЫ:
                    "что такое свойство квадрата" -> "что такое свойство числа a * a"
                    """).build();



    public static final String ENRICHED_QUESTION = "ENRICHED_QUESTION";
    public static final String ORIGINAL_QUESTION = "ORIGINAL_QUESTION";
    public static final String EXPANSION_RATIO = "EXPANSION_RATIO";

    private ChatClient chatClient;

    private final ChatModel chatModel;

    public static ExpansionQueryAdvisorBuilder builder(ChatModel chatModel1) {
        return new ExpansionQueryAdvisorBuilder().chatClient(ChatClient.builder(chatModel1)
                        .defaultOptions(OllamaOptions.builder()
                                .temperature(0.0)
                                .topK(1)
                                .topP(0.1)
                                .repeatPenalty(1.0)
                                .build()
                        )
                .build());
    }

    @Getter
    private final int order;

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
                .context(EXPANSION_RATIO, ratio)

                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain) {
        return chatClientResponse;
    }


}
