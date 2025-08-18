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
						SimpleLoggerAdvisor.builder().build()
//				, getRagAdviser()
				)
				.build();
	}

	private Advisor getRagAdviser() {
		return QuestionAnswerAdvisor.builder(vectorStore).promptTemplate(MY_PROMPT_TEMPLATE).searchRequest(
				SearchRequest.builder().topK(4).build()
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
