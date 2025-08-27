# Spring Function Tools 工作原理详解

## 1. Tools 注册过程

### 1.1 注册机制
Spring Function中的tools通过`@Bean`注解或函数式编程方式注册到Spring容器中，然后通过`FunctionCatalog`进行管理。

### 1.2 注册示例
```java
@Configuration
public class ToolConfiguration {
    
    // 方式1：通过@Bean注册
    @Bean
    public Function<WeatherRequest, WeatherResponse> weatherTool() {
        return request -> {
            // 模拟天气查询逻辑
            return new WeatherResponse(request.getCity(), "晴天", 25);
        };
    }
    
    // 方式2：通过函数式接口注册
    @Bean
    public Function<CalculateRequest, CalculateResponse> calculatorTool() {
        return request -> {
            double result = switch (request.getOperation()) {
                case "add" -> request.getA() + request.getB();
                case "subtract" -> request.getA() - request.getB();
                case "multiply" -> request.getA() * request.getB();
                case "divide" -> request.getA() / request.getB();
                default -> 0;
            };
            return new CalculateResponse(result);
        };
    }
}

// 请求和响应类
public record WeatherRequest(String city) {}
public record WeatherResponse(String city, String weather, int temperature) {}
public record CalculateRequest(double a, double b, String operation) {}
public record CalculateResponse(double result) {}
```

### 1.3 注册原理
- Spring容器启动时扫描所有`Function`类型的Bean
- `FunctionCatalog`将这些函数按名称进行索引
- 每个函数都会被包装成`FunctionInvocationWrapper`
- 支持反射机制获取函数的输入输出类型信息

## 2. Tools 调用过程

### 2.1 调用触发机制
当AI模型需要使用工具时，会生成tool call请求，Spring Function通过以下步骤处理：

```java
@RestController
public class ChatController {
    
    @Autowired
    private FunctionCatalog functionCatalog;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @PostMapping("/chat")
    public ChatResponse chat(@RequestBody ChatRequest request) {
        // 1. 处理用户消息，检测是否需要tool call
        List<Message> messages = processMessages(request.getMessages());
        
        // 2. 如果包含tool call，执行工具调用
        for (Message message : messages) {
            if (message.getToolCalls() != null) {
                for (ToolCall toolCall : message.getToolCalls()) {
                    Object result = invokeTool(toolCall);
                    // 将结果添加到消息链中
                    messages.add(createToolResultMessage(toolCall.getId(), result));
                }
            }
        }
        
        // 3. 将所有消息发送给AI模型生成最终响应
        return generateFinalResponse(messages);
    }
    
    private Object invokeTool(ToolCall toolCall) {
        try {
            // 从FunctionCatalog获取函数
            Function<Object, Object> function = functionCatalog.lookup(toolCall.getFunction().getName());
            
            // 解析参数
            Object arguments = parseArguments(toolCall.getFunction().getArguments(), function);
            
            // 执行函数
            return function.apply(arguments);
        } catch (Exception e) {
            return new ErrorResponse("Tool execution failed: " + e.getMessage());
        }
    }
}
```

### 2.2 参数解析和类型转换
```java
private Object parseArguments(String argumentsJson, Function<Object, Object> function) {
    try {
        // 获取函数的输入类型
        Type inputType = getFunctionInputType(function);
        
        // 将JSON参数转换为对应的Java对象
        return objectMapper.readValue(argumentsJson, objectMapper.constructType(inputType));
    } catch (Exception e) {
        throw new RuntimeException("Failed to parse arguments", e);
    }
}

private Type getFunctionInputType(Function<Object, Object> function) {
    // 通过反射获取函数的泛型类型信息
    return ((ParameterizedType) function.getClass().getGenericInterfaces()[0])
            .getActualTypeArguments()[0];
}
```

## 3. 核心技术原理

### 3.1 反射机制
Spring Function大量使用反射来：
- 动态获取函数的输入输出类型
- 运行时调用函数
- 进行参数类型检查和转换

```java
public class FunctionInvocationWrapper {
    private final Function<Object, Object> target;
    private final Type inputType;
    private final Type outputType;
    
    public FunctionInvocationWrapper(Function<Object, Object> target) {
        this.target = target;
        this.inputType = extractInputType(target);
        this.outputType = extractOutputType(target);
    }
    
    public Object invoke(Object input) {
        // 类型检查
        if (!isCompatibleType(input, inputType)) {
            throw new IllegalArgumentException("Input type mismatch");
        }
        
        // 执行函数
        Object result = target.apply(input);
        
        // 输出类型验证
        if (!isCompatibleType(result, outputType)) {
            throw new RuntimeException("Output type mismatch");
        }
        
        return result;
    }
}
```

### 3.2 JSON序列化/反序列化
```java
@Component
public class ToolCallProcessor {
    
    @Autowired
    private ObjectMapper objectMapper;
    
    public String serializeToolResult(Object result) {
        try {
            return objectMapper.writeValueAsString(result);
        } catch (JsonProcessingException e) {
            return "{\"error\":\"Serialization failed\"}";
        }
    }
    
    public <T> T deserializeToolArguments(String json, Class<T> targetType) {
        try {
            return objectMapper.readValue(json, targetType);
        } catch (Exception e) {
            throw new RuntimeException("Deserialization failed", e);
        }
    }
}
```

### 3.3 函数式编程模型
Spring Function基于Java 8+的函数式接口：
```java
// 基本函数类型
Function<Input, Output>     // 标准函数
Consumer<Input>            // 只有输入，无返回值
Supplier<Output>           // 无输入，只有返回值
BiFunction<Input1, Input2, Output>  // 两个输入参数

// 异步函数支持
Function<Input, Mono<Output>>     // 响应式编程支持
Function<Input, Flux<Output>>     // 流式处理支持
```

## 4. 调用结果传递给AI模型

### 4.1 结果封装
```java
public class ToolResultMessage {
    private String role = "tool";
    private String toolCallId;
    private String content;
    
    public ToolResultMessage(String toolCallId, Object result) {
        this.toolCallId = toolCallId;
        this.content = serializeResult(result);
    }
    
    private String serializeResult(Object result) {
        if (result instanceof String) {
            return (String) result;
        }
        // 将复杂对象转换为JSON字符串
        return objectMapper.writeValueAsString(result);
    }
}
```

### 4.2 完整的消息流处理
```java
@Service
public class ConversationService {
    
    public ChatResponse processConversation(List<Message> messages) {
        List<Message> processedMessages = new ArrayList<>(messages);
        
        // 处理所有tool calls
        boolean hasToolCalls = true;
        while (hasToolCalls) {
            hasToolCalls = false;
            
            for (Message message : new ArrayList<>(processedMessages)) {
                if (message.getToolCalls() != null) {
                    hasToolCalls = true;
                    
                    // 执行每个tool call
                    for (ToolCall toolCall : message.getToolCalls()) {
                        Object result = executeTool(toolCall);
                        
                        // 添加tool结果消息
                        processedMessages.add(new ToolResultMessage(
                            toolCall.getId(), 
                            result
                        ));
                    }
                    
                    // 发送给AI模型获取下一步响应
                    Message aiResponse = callAIModel(processedMessages);
                    processedMessages.add(aiResponse);
                    break;
                }
            }
        }
        
        return new ChatResponse(processedMessages);
    }
}
```

### 4.3 与AI模型的集成示例
```java
@Service
public class AIIntegrationService {
    
    @Autowired
    private OpenAIClient openAIClient;
    
    public Message callAIWithToolResults(List<Message> messages) {
        // 构建OpenAI API请求
        ChatCompletionRequest request = ChatCompletionRequest.builder()
            .model("gpt-4")
            .messages(convertToOpenAIMessages(messages))
            .tools(getAvailableTools())  // 告诉AI可用的工具
            .toolChoice("auto")          // 让AI自动决定是否使用工具
            .build();
            
        ChatCompletionResponse response = openAIClient.createChatCompletion(request);
        
        return convertToMessage(response.getChoices().get(0).getMessage());
    }
    
    private List<Tool> getAvailableTools() {
        // 从FunctionCatalog获取所有可用工具的描述
        return functionCatalog.getNames().stream()
            .map(this::createToolDescription)
            .collect(Collectors.toList());
    }
}
```

## 5. 完整示例：天气查询工具

### 5.1 工具定义和注册
```java
@Component
public class WeatherService {
    
    @Bean
    public Function<WeatherQuery, WeatherInfo> weatherTool() {
        return query -> {
            // 模拟天气API调用
            return new WeatherInfo(
                query.city(),
                getCurrentWeather(query.city()),
                getCurrentTemperature(query.city())
            );
        };
    }
    
    private String getCurrentWeather(String city) {
        // 实际实现中这里会调用天气API
        return "晴天";
    }
    
    private int getCurrentTemperature(String city) {
        return 25; // 摄氏度
    }
}

public record WeatherQuery(String city) {}
public record WeatherInfo(String city, String condition, int temperature) {}
```

### 5.2 工具调用流程
```java
// 1. 用户输入："北京今天天气怎么样？"

// 2. AI模型生成tool call
{
    "role": "assistant",
    "content": null,
    "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "weatherTool",
            "arguments": "{\"city\":\"北京\"}"
        }
    }]
}

// 3. Spring Function执行工具
WeatherQuery query = new WeatherQuery("北京");
WeatherInfo result = weatherTool.apply(query);
// result = WeatherInfo(city="北京", condition="晴天", temperature=25)

// 4. 工具结果消息
{
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "{\"city\":\"北京\",\"condition\":\"晴天\",\"temperature\":25}"
}

// 5. AI模型根据工具结果生成最终回答
{
    "role": "assistant",
    "content": "根据查询结果，北京今天是晴天，气温25摄氏度，天气很不错！"
}
```

## 总结

Spring Function的tools机制通过以下关键技术实现：

1. **注册阶段**：利用Spring IoC容器和函数式编程模型
2. **调用阶段**：通过反射、JSON序列化和类型安全机制
3. **结果传递**：采用标准的消息格式与AI模型集成
4. **核心原理**：函数式编程 + 反射 + JSON处理 + 消息驱动架构

这种设计使得工具的定义、注册和使用都非常灵活和类型安全，同时保持了与各种AI模型的良好集成能力。