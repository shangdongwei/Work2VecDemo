# Spring AI框架详细教程 - 从入门到工具调用

## 1. Spring AI 框架概述

### 1.1 什么是Spring AI？
Spring AI是Spring官方推出的人工智能集成框架，它让Java开发者可以轻松地在Spring应用中集成各种AI模型，包括：
- OpenAI GPT系列
- Azure OpenAI
- Google Vertex AI
- 阿里云通义千问
- 本地开源模型等

### 1.2 核心设计理念
```
用户请求 → Spring AI → AI模型 → 响应处理 → 返回结果
```

Spring AI提供了统一的抽象层，让你可以轻松切换不同的AI提供商，而不需要修改业务代码。

### 1.3 主要组件结构
```
spring-ai-core              # 核心抽象接口
├── ChatClient              # 聊天客户端
├── ChatModel               # 聊天模型抽象
├── EmbeddingClient         # 向量嵌入客户端
├── ImageClient             # 图像生成客户端
└── VectorStore             # 向量数据库

spring-ai-openai            # OpenAI实现
spring-ai-azure-openai      # Azure OpenAI实现
spring-ai-ollama            # Ollama本地模型实现
spring-ai-function          # 工具函数调用
```

## 2. Spring AI 基础使用示例

### 2.1 Maven依赖配置
```xml
<dependencies>
    <!-- Spring AI核心依赖 -->
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
        <version>1.0.0-M3</version>
    </dependency>
    
    <!-- Spring Boot Web -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
</dependencies>
```

### 2.2 配置文件
```yaml
# application.yml
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}  # 从环境变量获取
      base-url: https://api.openai.com
      chat:
        options:
          model: gpt-3.5-turbo
          temperature: 0.7
          max-tokens: 1000
```

### 2.3 最简单的聊天示例
```java
@RestController
public class SimpleChatController {
    
    private final ChatClient chatClient;
    
    public SimpleChatController(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder.build();
    }
    
    @GetMapping("/chat")
    public String chat(@RequestParam String message) {
        return chatClient.prompt()
            .user(message)
            .call()
            .content();
    }
}
```

## 3. Spring AI 核心架构原理

### 3.1 ChatClient 工作原理

```java
// ChatClient的内部结构简化版
public class ChatClient {
    
    private final ChatModel chatModel;        // AI模型接口
    private final List<Function> functions;   // 可用工具函数
    private final ConversationMemory memory;  // 对话记忆
    
    public ChatResponse call(ChatRequest request) {
        // 1. 预处理用户输入
        List<Message> messages = preprocessMessages(request);
        
        // 2. 检查是否需要工具调用
        if (needsToolCall(messages)) {
            return handleToolCall(messages);
        }
        
        // 3. 直接调用AI模型
        return chatModel.call(new Prompt(messages));
    }
    
    private ChatResponse handleToolCall(List<Message> messages) {
        ChatResponse response = chatModel.call(new Prompt(messages));
        
        // 如果AI返回了tool calls
        if (response.getResult().getOutput().getToolCalls() != null) {
            // 执行工具调用
            List<Message> toolResults = executeTools(response.getResult().getOutput().getToolCalls());
            
            // 将工具结果添加到对话中
            messages.addAll(toolResults);
            
            // 再次调用AI模型生成最终回答
            return chatModel.call(new Prompt(messages));
        }
        
        return response;
    }
}
```

### 3.2 消息流处理机制

Spring AI使用消息链来管理对话上下文：

```java
public class MessageChain {
    private List<Message> messages = new ArrayList<>();
    
    // 消息类型
    public enum MessageType {
        USER,        // 用户消息
        ASSISTANT,   // AI助手消息  
        SYSTEM,      // 系统消息
        TOOL         // 工具调用结果消息
    }
    
    public void addUserMessage(String content) {
        messages.add(new UserMessage(content));
    }
    
    public void addAssistantMessage(String content, List<ToolCall> toolCalls) {
        messages.add(new AssistantMessage(content, toolCalls));
    }
    
    public void addToolResultMessage(String toolCallId, String result) {
        messages.add(new ToolMessage(toolCallId, result));
    }
}
```

## 4. 工具函数调用详细原理

### 4.1 什么是Function Calling？

Function Calling（工具调用）是让AI模型能够调用外部函数来获取实时数据或执行特定操作的机制。

```
用户: "北京今天天气怎么样？"
  ↓
AI模型: "我需要查询天气信息"
  ↓
生成工具调用: getWeather(city="北京")
  ↓  
Spring AI执行工具函数
  ↓
返回结果: {"city":"北京", "weather":"晴", "temperature":25}
  ↓
AI模型: "根据查询结果，北京今天天气晴朗，气温25度"
```

### 4.2 工具函数的定义和注册

#### 方式1：使用@Bean注解
```java
@Configuration
public class ToolConfiguration {
    
    @Bean
    @Description("获取指定城市的天气信息")  // AI模型会看到这个描述
    public Function<WeatherRequest, WeatherResponse> weatherFunction() {
        return request -> {
            // 实际开发中这里会调用天气API
            System.out.println("正在查询 " + request.city() + " 的天气...");
            
            // 模拟API调用延迟
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
            // 模拟天气数据
            return new WeatherResponse(
                request.city(),
                "晴天",
                25,
                "适合出行的好天气"
            );
        };
    }
    
    @Bean
    @Description("执行数学计算")
    public Function<CalculatorRequest, CalculatorResponse> calculatorFunction() {
        return request -> {
            double result = switch (request.operation()) {
                case "add" -> request.a() + request.b();
                case "subtract" -> request.a() - request.b();
                case "multiply" -> request.a() * request.b();
                case "divide" -> request.b() != 0 ? request.a() / request.b() : 0;
                default -> 0;
            };
            
            return new CalculatorResponse(
                request.a(),
                request.b(), 
                request.operation(),
                result
            );
        };
    }
}

// 数据传输对象（DTO）
public record WeatherRequest(
    @JsonProperty("city") String city
) {}

public record WeatherResponse(
    String city,
    String weather,
    int temperature,
    String description
) {}

public record CalculatorRequest(
    @JsonProperty("a") double a,
    @JsonProperty("b") double b,
    @JsonProperty("operation") String operation
) {}

public record CalculatorResponse(
    double a,
    double b,
    String operation,
    double result
) {}
```

#### 方式2：实现FunctionCallback接口
```java
@Component("weatherTool")
public class WeatherFunctionCallback implements FunctionCallback {
    
    @Override
    public String getName() {
        return "get_weather";
    }
    
    @Override
    public String getDescription() {
        return "获取指定城市的实时天气信息";
    }
    
    @Override
    public String getInputTypeSchema() {
        // 定义输入参数的JSON Schema
        return """
            {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
            """;
    }
    
    @Override
    public String call(String functionArguments) {
        try {
            // 解析输入参数
            ObjectMapper mapper = new ObjectMapper();
            JsonNode args = mapper.readTree(functionArguments);
            String city = args.get("city").asText();
            
            // 执行业务逻辑
            WeatherInfo weather = getWeatherFromAPI(city);
            
            // 返回JSON格式的结果
            return mapper.writeValueAsString(weather);
            
        } catch (Exception e) {
            return "{\"error\": \"" + e.getMessage() + "\"}";
        }
    }
    
    private WeatherInfo getWeatherFromAPI(String city) {
        // 这里实际会调用天气API
        return new WeatherInfo(city, "晴天", 25);
    }
}
```

### 4.3 工具调用的完整Controller示例

```java
@RestController
@RequestMapping("/api/chat")
public class ChatWithToolsController {
    
    private final ChatClient chatClient;
    private final Logger logger = LoggerFactory.getLogger(ChatWithToolsController.class);
    
    public ChatWithToolsController(
            ChatClient.Builder chatClientBuilder,
            Function<WeatherRequest, WeatherResponse> weatherFunction,
            Function<CalculatorRequest, CalculatorResponse> calculatorFunction) {
        
        this.chatClient = chatClientBuilder
            .defaultSystem("你是一个有用的AI助手，可以查询天气和进行数学计算")
            .defaultFunctions("weatherFunction", "calculatorFunction")  // 注册可用工具
            .build();
    }
    
    @PostMapping("/message")
    public ChatResponse chat(@RequestBody ChatRequest request) {
        logger.info("收到用户消息: {}", request.getMessage());
        
        try {
            ChatResponse response = chatClient
                .prompt()
                .user(request.getMessage())
                .call()
                .chatResponse();
                
            logger.info("AI回复: {}", response.getResult().getOutput().getContent());
            return response;
            
        } catch (Exception e) {
            logger.error("处理聊天消息时出错", e);
            throw new RuntimeException("聊天服务暂时不可用", e);
        }
    }
    
    @PostMapping("/stream")
    public Flux<String> streamChat(@RequestBody ChatRequest request) {
        return chatClient
            .prompt()
            .user(request.getMessage())
            .stream()
            .content();
    }
}

// 请求响应对象
public record ChatRequest(String message) {}
```

### 4.4 工具调用的内部执行流程

让我们深入了解工具调用的内部机制：

```java
// 简化的工具调用执行器
@Component
public class FunctionCallExecutor {
    
    private final ApplicationContext applicationContext;
    private final ObjectMapper objectMapper;
    
    public FunctionCallExecutor(ApplicationContext applicationContext) {
        this.applicationContext = applicationContext;
        this.objectMapper = new ObjectMapper();
    }
    
    public Object executeFunction(String functionName, String arguments) {
        try {
            // 1. 从Spring容器中获取函数Bean
            Object functionBean = applicationContext.getBean(functionName);
            
            if (functionBean instanceof Function<?, ?> function) {
                // 2. 获取函数的输入类型
                Type inputType = getFunctionInputType(function);
                
                // 3. 将JSON参数反序列化为Java对象
                Object input = objectMapper.readValue(arguments, 
                    objectMapper.constructType(inputType));
                
                // 4. 执行函数
                Object result = ((Function<Object, Object>) function).apply(input);
                
                // 5. 序列化返回结果
                return result;
                
            } else if (functionBean instanceof FunctionCallback callback) {
                // 处理FunctionCallback类型的工具
                return callback.call(arguments);
            }
            
        } catch (Exception e) {
            logger.error("执行函数 {} 时出错: {}", functionName, e.getMessage());
            return new ErrorResult("函数执行失败: " + e.getMessage());
        }
        
        return new ErrorResult("未找到函数: " + functionName);
    }
    
    // 通过反射获取Function的输入类型
    private Type getFunctionInputType(Function<?, ?> function) {
        Type[] interfaces = function.getClass().getGenericInterfaces();
        for (Type iface : interfaces) {
            if (iface instanceof ParameterizedType paramType) {
                if (paramType.getRawType().equals(Function.class)) {
                    return paramType.getActualTypeArguments()[0];
                }
            }
        }
        return Object.class;
    }
}
```

### 4.5 详细的消息流处理

```java
@Service
public class ConversationService {
    
    private final ChatModel chatModel;
    private final FunctionCallExecutor functionExecutor;
    
    public ChatResponse processConversationWithTools(String userMessage) {
        List<Message> messages = new ArrayList<>();
        messages.add(new UserMessage(userMessage));
        
        int maxIterations = 10;  // 防止无限循环
        int iteration = 0;
        
        while (iteration < maxIterations) {
            iteration++;
            
            // 调用AI模型
            Prompt prompt = new Prompt(messages);
            ChatResponse response = chatModel.call(prompt);
            
            AssistantMessage assistantMessage = response.getResult().getOutput();
            messages.add(assistantMessage);
            
            // 检查是否有工具调用
            List<ToolCall> toolCalls = assistantMessage.getToolCalls();
            if (toolCalls == null || toolCalls.isEmpty()) {
                // 没有工具调用，返回最终结果
                return response;
            }
            
            // 执行所有工具调用
            for (ToolCall toolCall : toolCalls) {
                logger.info("执行工具: {} 参数: {}", 
                    toolCall.getName(), toolCall.getArguments());
                
                Object result = functionExecutor.executeFunction(
                    toolCall.getName(), 
                    toolCall.getArguments()
                );
                
                // 将工具结果添加到消息历史
                messages.add(new ToolResponseMessage(
                    toolCall.getId(),
                    objectMapper.writeValueAsString(result)
                ));
            }
        }
        
        throw new RuntimeException("工具调用达到最大迭代次数");
    }
}
```

## 5. 完整的工作示例

### 5.1 主应用类
```java
@SpringBootApplication
public class SpringAiDemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(SpringAiDemoApplication.class, args);
    }
}
```

### 5.2 测试用的Web页面
```html
<!DOCTYPE html>
<html>
<head>
    <title>Spring AI 聊天测试</title>
    <meta charset="UTF-8">
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <input type="text" id="user-input" placeholder="输入消息..." />
        <button onclick="sendMessage()">发送</button>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            // 显示用户消息
            addMessage('用户', message);
            input.value = '';
            
            try {
                const response = await fetch('/api/chat/message', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                addMessage('AI助手', data.result.output.content);
                
            } catch (error) {
                addMessage('系统', '发生错误: ' + error.message);
            }
        }
        
        function addMessage(sender, content) {
            const messages = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.innerHTML = `<strong>${sender}:</strong> ${content}`;
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        // 回车发送消息
        document.getElementById('user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
```

### 5.3 测试对话示例

**用户**: 北京今天天气怎么样？

**AI助手**: *（内部调用weatherFunction）*
根据查询结果，北京今天天气晴朗，气温25度，是适合出行的好天气！

**用户**: 帮我算一下25乘以8等于多少

**AI助手**: *（内部调用calculatorFunction）*
25乘以8等于200。

**用户**: 如果北京今天的温度乘以8，结果是多少？

**AI助手**: *（内部先调用weatherFunction获取温度，再调用calculatorFunction计算）*
根据查询，北京今天的温度是25度，25乘以8等于200。

## 6. 高级特性和最佳实践

### 6.1 异步工具调用
```java
@Bean
public Function<String, Mono<String>> asyncWeatherFunction() {
    return city -> {
        return WebClient.create()
            .get()
            .uri("https://api.weather.com/v1/current?city={city}", city)
            .retrieve()
            .bodyToMono(String.class)
            .timeout(Duration.ofSeconds(5))
            .onErrorReturn("天气查询失败");
    };
}
```

### 6.2 工具调用的错误处理
```java
@Bean
public Function<WeatherRequest, WeatherResponse> robustWeatherFunction() {
    return request -> {
        try {
            // 验证输入
            if (request.city() == null || request.city().trim().isEmpty()) {
                return new WeatherResponse("", "错误", 0, "城市名称不能为空");
            }
            
            // 执行业务逻辑
            return callWeatherAPI(request.city());
            
        } catch (Exception e) {
            logger.error("天气查询失败", e);
            return new WeatherResponse(
                request.city(), 
                "查询失败", 
                0, 
                "暂时无法获取天气信息: " + e.getMessage()
            );
        }
    };
}
```

### 6.3 工具调用的日志监控
```java
@Component
public class FunctionCallInterceptor {
    
    @EventListener
    public void onFunctionCall(FunctionCallEvent event) {
        logger.info("工具调用开始: 函数={}, 参数={}, 用户={}", 
            event.getFunctionName(),
            event.getArguments(),
            event.getUserId()
        );
    }
    
    @EventListener  
    public void onFunctionCallComplete(FunctionCallCompleteEvent event) {
        logger.info("工具调用完成: 函数={}, 耗时={}ms, 结果={}", 
            event.getFunctionName(),
            event.getDurationMs(),
            event.getResult()
        );
    }
}
```

## 7. 总结

Spring AI的工具调用机制通过以下关键步骤实现：

1. **工具注册**: 通过Spring Bean容器管理Function实例
2. **AI推理**: ChatModel分析用户请求，决定是否需要调用工具
3. **参数解析**: 将JSON参数反序列化为Java对象
4. **工具执行**: 通过反射机制调用对应的Function
5. **结果处理**: 将执行结果序列化后返回给AI模型
6. **响应生成**: AI模型基于工具结果生成最终的自然语言回答

这种设计让Java开发者能够：
- 轻松定义和注册工具函数
- 享受类型安全的好处
- 灵活处理复杂的业务逻辑
- 与Spring生态系统无缝集成

通过Spring AI，你可以快速构建具有实时数据获取能力的智能应用！