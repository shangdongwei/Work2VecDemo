import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.ArrayList;

/**
 * JIRA工单创建工具
 * 支持创建Tasks, User Stories, Epics 和 Features
 * 兼容ALM和WPB两种JIRA类型
 */
public class JiraTicketCreator {
    
    private final String baseUrl;
    private final String username;
    private final String password;
    private final String authHeader;
    private final ObjectMapper objectMapper;
    private final CloseableHttpClient httpClient;
    
    // JIRA工单类型枚举
    public enum TicketType {
        TASK("Task"),
        USER_STORY("Story"), 
        EPIC("Epic"),
        FEATURE("Feature");
        
        private final String jiraTypeName;
        
        TicketType(String jiraTypeName) {
            this.jiraTypeName = jiraTypeName;
        }
        
        public String getJiraTypeName() {
            return jiraTypeName;
        }
    }
    
    // JIRA类型枚举
    public enum JiraType {
        ALM, WPB
    }
    
    // 工单创建请求类
    public static class TicketRequest {
        private String projectKey;
        private TicketType ticketType;
        private String summary;
        private String description;
        private String reporter;
        private String sprintId;
        private Integer storyPoints;
        private String parentKey; // 用于Task的parent link
        private JiraType jiraType;
        
        // 构造函数和getter/setter方法
        public TicketRequest(String projectKey, TicketType ticketType, String summary) {
            this.projectKey = projectKey;
            this.ticketType = ticketType;
            this.summary = summary;
        }
        
        // Getter和Setter方法
        public String getProjectKey() { return projectKey; }
        public void setProjectKey(String projectKey) { this.projectKey = projectKey; }
        
        public TicketType getTicketType() { return ticketType; }
        public void setTicketType(TicketType ticketType) { this.ticketType = ticketType; }
        
        public String getSummary() { return summary; }
        public void setSummary(String summary) { this.summary = summary; }
        
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        
        public String getReporter() { return reporter; }
        public void setReporter(String reporter) { this.reporter = reporter; }
        
        public String getSprintId() { return sprintId; }
        public void setSprintId(String sprintId) { this.sprintId = sprintId; }
        
        public Integer getStoryPoints() { return storyPoints; }
        public void setStoryPoints(Integer storyPoints) { this.storyPoints = storyPoints; }
        
        public String getParentKey() { return parentKey; }
        public void setParentKey(String parentKey) { this.parentKey = parentKey; }
        
        public JiraType getJiraType() { return jiraType; }
        public void setJiraType(JiraType jiraType) { this.jiraType = jiraType; }
    }
    
    // 响应类
    public static class TicketResponse {
        private boolean success;
        private String issueKey;
        private String message;
        
        public TicketResponse(boolean success, String issueKey, String message) {
            this.success = success;
            this.issueKey = issueKey;
            this.message = message;
        }
        
        public boolean isSuccess() { return success; }
        public String getIssueKey() { return issueKey; }
        public String getMessage() { return message; }
    }
    
    public JiraTicketCreator(String baseUrl, String username, String password) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.username = username;
        this.password = password;
        this.authHeader = "Basic " + Base64.getEncoder()
            .encodeToString((username + ":" + password).getBytes(StandardCharsets.UTF_8));
        this.objectMapper = new ObjectMapper();
        this.httpClient = HttpClients.createDefault();
    }
    
    /**
     * 创建JIRA工单
     */
    public TicketResponse createTicket(TicketRequest request) {
        try {
            // 如果是Feature类型，先检查项目是否支持Feature配置
            if (request.getTicketType() == TicketType.FEATURE) {
                if (!checkFeatureSupport(request.getProjectKey())) {
                    return new TicketResponse(false, null, 
                        "Sorry, the selected project does not have this configuration.");
                }
            }
            
            // 构建创建工单的JSON请求体
            ObjectNode createRequest = buildCreateRequest(request);
            
            // 发送创建请求
            HttpPost httpPost = new HttpPost(baseUrl + "/rest/api/3/issue");
            httpPost.setHeader("Authorization", authHeader);
            httpPost.setHeader("Content-Type", "application/json");
            httpPost.setEntity(new StringEntity(createRequest.toString(), StandardCharsets.UTF_8));
            
            HttpResponse response = httpClient.execute(httpPost);
            String responseBody = EntityUtils.toString(response.getEntity());
            
            if (response.getStatusLine().getStatusCode() == 201) {
                JsonNode responseJson = objectMapper.readTree(responseBody);
                String issueKey = responseJson.get("key").asText();
                return new TicketResponse(true, issueKey, 
                    "Successfully created " + request.getTicketType().name() + ": " + issueKey);
            } else {
                JsonNode errorJson = objectMapper.readTree(responseBody);
                String errorMessage = extractErrorMessage(errorJson);
                return new TicketResponse(false, null, 
                    "Failed to create ticket: " + errorMessage);
            }
            
        } catch (Exception e) {
            return new TicketResponse(false, null, 
                "Exception occurred while creating ticket: " + e.getMessage());
        }
    }
    
    /**
     * 检查项目是否支持Feature类型
     */
    private boolean checkFeatureSupport(String projectKey) {
        try {
            // 获取项目的工单类型配置
            HttpGet httpGet = new HttpGet(baseUrl + "/rest/api/3/project/" + projectKey);
            httpGet.setHeader("Authorization", authHeader);
            httpGet.setHeader("Content-Type", "application/json");
            
            HttpResponse response = httpClient.execute(httpGet);
            
            if (response.getStatusLine().getStatusCode() != 200) {
                return false;
            }
            
            String responseBody = EntityUtils.toString(response.getEntity());
            JsonNode projectJson = objectMapper.readTree(responseBody);
            
            // 获取项目的工单类型
            JsonNode issueTypes = projectJson.get("issueTypes");
            if (issueTypes != null && issueTypes.isArray()) {
                for (JsonNode issueType : issueTypes) {
                    String typeName = issueType.get("name").asText();
                    if ("Feature".equalsIgnoreCase(typeName)) {
                        return true;
                    }
                }
            }
            
            return false;
            
        } catch (Exception e) {
            System.err.println("Error checking feature support: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 构建创建工单的请求体
     */
    private ObjectNode buildCreateRequest(TicketRequest request) {
        ObjectNode createRequest = objectMapper.createObjectNode();
        ObjectNode fields = objectMapper.createObjectNode();
        
        // 基本字段
        ObjectNode project = objectMapper.createObjectNode();
        project.put("key", request.getProjectKey());
        fields.set("project", project);
        
        ObjectNode issuetype = objectMapper.createObjectNode();
        issuetype.put("name", request.getTicketType().getJiraTypeName());
        fields.set("issuetype", issuetype);
        
        fields.put("summary", request.getSummary());
        
        if (request.getDescription() != null && !request.getDescription().isEmpty()) {
            fields.put("description", request.getDescription());
        }
        
        // Reporter字段
        if (request.getReporter() != null && !request.getReporter().isEmpty()) {
            ObjectNode reporter = objectMapper.createObjectNode();
            reporter.put("name", request.getReporter());
            fields.set("reporter", reporter);
        }
        
        // Sprint字段（根据JIRA类型调整字段名称）
        if (request.getSprintId() != null && !request.getSprintId().isEmpty()) {
            String sprintFieldName = getSprintFieldName(request.getJiraType());
            if (sprintFieldName != null) {
                ArrayNode sprintArray = objectMapper.createArrayNode();
                sprintArray.add(Integer.parseInt(request.getSprintId()));
                fields.set(sprintFieldName, sprintArray);
            }
        }
        
        // Story Points字段（根据JIRA类型调整字段名称）
        if (request.getStoryPoints() != null && request.getStoryPoints() > 0) {
            String storyPointsFieldName = getStoryPointsFieldName(request.getJiraType());
            if (storyPointsFieldName != null) {
                fields.put(storyPointsFieldName, request.getStoryPoints());
            }
        }
        
        // Parent Link字段（用于Task的父级关联）
        if (request.getParentKey() != null && !request.getParentKey().isEmpty()) {
            ObjectNode parent = objectMapper.createObjectNode();
            parent.put("key", request.getParentKey());
            fields.set("parent", parent);
        }
        
        createRequest.set("fields", fields);
        return createRequest;
    }
    
    /**
     * 根据JIRA类型获取Sprint字段名称
     */
    private String getSprintFieldName(JiraType jiraType) {
        if (jiraType == null) return "customfield_10020"; // 默认Sprint字段
        
        switch (jiraType) {
            case ALM:
                return "customfield_10020"; // ALM中的Sprint字段ID
            case WPB:
                return "customfield_10021"; // WPB中的Sprint字段ID
            default:
                return "customfield_10020";
        }
    }
    
    /**
     * 根据JIRA类型获取Story Points字段名称
     */
    private String getStoryPointsFieldName(JiraType jiraType) {
        if (jiraType == null) return "customfield_10016"; // 默认Story Points字段
        
        switch (jiraType) {
            case ALM:
                return "customfield_10016"; // ALM中的Story Points字段ID
            case WPB:
                return "customfield_10017"; // WPB中的Story Points字段ID
            default:
                return "customfield_10016";
        }
    }
    
    /**
     * 提取错误信息
     */
    private String extractErrorMessage(JsonNode errorJson) {
        try {
            if (errorJson.has("errorMessages") && errorJson.get("errorMessages").isArray()) {
                ArrayNode errorMessages = (ArrayNode) errorJson.get("errorMessages");
                if (errorMessages.size() > 0) {
                    return errorMessages.get(0).asText();
                }
            }
            
            if (errorJson.has("errors")) {
                JsonNode errors = errorJson.get("errors");
                if (errors.isObject()) {
                    return errors.toString();
                }
            }
            
            return errorJson.toString();
        } catch (Exception e) {
            return "Unknown error occurred";
        }
    }
    
    /**
     * 批量创建工单
     */
    public List<TicketResponse> createMultipleTickets(List<TicketRequest> requests) {
        List<TicketResponse> responses = new ArrayList<>();
        
        for (TicketRequest request : requests) {
            TicketResponse response = createTicket(request);
            responses.add(response);
            
            // 如果创建失败，可以选择是否继续创建其他工单
            if (!response.isSuccess()) {
                System.err.println("Failed to create ticket: " + response.getMessage());
            }
        }
        
        return responses;
    }
    
    /**
     * 关闭HTTP客户端
     */
    public void close() {
        try {
            if (httpClient != null) {
                httpClient.close();
            }
        } catch (Exception e) {
            System.err.println("Error closing HTTP client: " + e.getMessage());
        }
    }
    
    /**
     * 示例用法
     */
    public static void main(String[] args) {
        // 创建JIRA客户端
        JiraTicketCreator jiraClient = new JiraTicketCreator(
            "https://your-jira-instance.com",
            "your-username", 
            "your-password"
        );
        
        try {
            // 示例1: 创建Epic
            TicketRequest epicRequest = new TicketRequest("PROJ", TicketType.EPIC, "Sample Epic");
            epicRequest.setDescription("This is a sample epic");
            epicRequest.setReporter("reporter.user");
            epicRequest.setJiraType(JiraType.ALM);
            
            TicketResponse epicResponse = jiraClient.createTicket(epicRequest);
            System.out.println("Epic creation result: " + epicResponse.getMessage());
            
            // 示例2: 创建User Story（关联到Epic）
            if (epicResponse.isSuccess()) {
                TicketRequest storyRequest = new TicketRequest("PROJ", TicketType.USER_STORY, "Sample User Story");
                storyRequest.setDescription("This is a sample user story");
                storyRequest.setReporter("reporter.user");
                storyRequest.setStoryPoints(5);
                storyRequest.setSprintId("123");
                storyRequest.setParentKey(epicResponse.getIssueKey()); // 关联到Epic
                storyRequest.setJiraType(JiraType.ALM);
                
                TicketResponse storyResponse = jiraClient.createTicket(storyRequest);
                System.out.println("Story creation result: " + storyResponse.getMessage());
                
                // 示例3: 创建Task（关联到User Story）
                if (storyResponse.isSuccess()) {
                    TicketRequest taskRequest = new TicketRequest("PROJ", TicketType.TASK, "Sample Task");
                    taskRequest.setDescription("This is a sample task");
                    taskRequest.setReporter("reporter.user");
                    taskRequest.setSprintId("123");
                    taskRequest.setParentKey(storyResponse.getIssueKey()); // 关联到User Story
                    taskRequest.setJiraType(JiraType.ALM);
                    
                    TicketResponse taskResponse = jiraClient.createTicket(taskRequest);
                    System.out.println("Task creation result: " + taskResponse.getMessage());
                }
            }
            
            // 示例4: 尝试创建Feature（会检查项目配置）
            TicketRequest featureRequest = new TicketRequest("PROJ", TicketType.FEATURE, "Sample Feature");
            featureRequest.setDescription("This is a sample feature");
            featureRequest.setReporter("reporter.user");
            featureRequest.setStoryPoints(13);
            featureRequest.setJiraType(JiraType.WPB);
            
            TicketResponse featureResponse = jiraClient.createTicket(featureRequest);
            System.out.println("Feature creation result: " + featureResponse.getMessage());
            
        } finally {
            // 关闭客户端
            jiraClient.close();
        }
    }
}