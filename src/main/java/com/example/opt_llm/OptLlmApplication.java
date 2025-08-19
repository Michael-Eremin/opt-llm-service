package com.example.opt_llm;

import com.example.opt_llm.repo.ChatRepository;
import com.example.opt_llm.services.PostgresChatMemory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class OptLlmApplication {

	private static final PromptTemplate MY_PROMPT_TEMPLATE = new PromptTemplate(
			"{query}\n\n" +
					"Контекст:\n" +
					"---------------------\n" +
					"{question_answer_context}\n" +
					"---------------------\n\n" +
					"Отвечай только на основе контекста выше. Если информации нет в контексте, сообщи, что не можешь ответить."
	);



	@Autowired
	private ChatRepository chatRepository;

	@Autowired
	private VectorStore vectorStore;


	@Bean
	public ChatClient chatClient(ChatClient.Builder builder) {
		return builder.defaultAdvisors(
				getHistoryAdvisor(),
						SimpleLoggerAdvisor.builder().build(),
						getRagAdviser(),
						SimpleLoggerAdvisor.builder().build()
				)
				.defaultOptions(
						OllamaOptions.builder()
//								Лучше 0.2 - 0.3 (чтобы сильно не изобретала)
								.temperature(0.3)
//								Выбираем с вероятностью соответствия токена
//								70% - в основном зайдет готовый токен около 70%
//								и мало мусора, чтобы добить нехватку.
//								Если взять больше (90%), то попадут
//								наиболее подходящие (около 70%) но и много мусора
//								добьется на оставшиеся 20%
								.topP(0.7)
//								количество подходящих токенов
//								сколько взять токенов
//								(должна быть вариативность, но небольшая)
								.topK(20)
//								Штраф за повтор - не большой, но должен быть
								.repeatPenalty(1.1)
								.build()
				)
				.build();
	}

	private Advisor getRagAdviser() {
		return QuestionAnswerAdvisor.builder(vectorStore).promptTemplate(MY_PROMPT_TEMPLATE).searchRequest(
				SearchRequest.builder()
//						Сколько взять чанков
//						если взять 40, то ждать ответ минут 10
						.topK(4)
//						как topP в chatClient для токенов, а этот для чанков
						.similarityThreshold(0.65)
						.build()
		).build();
	}


	private Advisor getHistoryAdvisor() {
		return MessageChatMemoryAdvisor.builder(getChatMemory()).order(-10).build();
	}

	private ChatMemory getChatMemory() {
		return PostgresChatMemory.builder()
				.maxMessages(4)
				.chatMemoryRepository(chatRepository)
				.build();
	}


	public static void main(String[] args) {
		ConfigurableApplicationContext context = SpringApplication.run(OptLlmApplication.class, args);
		ChatClient chatClient = context.getBean(ChatClient.class);
	}



}
