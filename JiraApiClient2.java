import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * JIRA Software Data Center API客户端
 * 支持查询、创建、更新Task和Story操作
 */
public class JiraApiClient {
    
    private final String baseUrl;
    private final HttpClient httpClient;
    private final String authorization;
    
    // 常用字段常量
    public static final String FIELD_SUMMARY = "summary";
    public static final String FIELD_DESCRIPTION = "description";
    public static final String FIELD_ISSUE_TYPE = "issuetype";
    public static final String FIELD_PRIORITY = "priority";
    public static final String FIELD_ASSIGNEE = "assignee";
    public static final String FIELD_REPORTER = "reporter";
    public static final String FIELD_PROJECT = "project";
    public static final String FIELD_LABELS = "labels";
    public static final String FIELD_COMPONENTS = "components";
    public static final String FIELD_FIX_VERSIONS = "fixVersions";
    public static final String FIELD_VERSIONS = "versions";
    public static final String FIELD_DUE_DATE = "duedate";
    public static final String FIELD_ENVIRONMENT = "environment";
    public static final String FIELD_STATUS = "status";
    public static final String FIELD_RESOLUTION = "resolution";
    
    // Issue类型常量
    public static final String ISSUE_TYPE_TASK = "Task";
    public static final String ISSUE_TYPE_STORY = "Story";
    public static final String ISSUE_TYPE_BUG = "Bug";
    
    public JiraApiClient(String baseUrl, String username, String apiToken) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();
        this.authorization = "Basic " + Base64.getEncoder()
                .encodeToString((username + ":" + apiToken).getBytes(StandardCharsets.UTF_8));
    }
    
    /**
     * 查询单个Issue
     * @param issueKeyOrId Issue Key或ID
     * @param options 查询选项
     * @return JSONObject Issue信息
     */
    public JSONObject getIssue(String issueKeyOrId, IssueQueryOptions options) throws Exception {
        StringBuilder url = new StringBuilder(baseUrl + "rest/api/latest/issue/" + issueKeyOrId);
        
        List<String> queryParams = new ArrayList<>();
        
        if (options.expand != null && !options.expand.isEmpty()) {
            queryParams.add("expand=" + String.join(",", options.expand));
        }
        
        if (options.fields != null && !options.fields.isEmpty()) {
            queryParams.add("fields=" + String.join(",", options.fields));
        }
        
        if (options.properties != null && !options.properties.isEmpty()) {
            queryParams.add("properties=" + String.join(",", options.properties));
        }
        
        if (options.fieldsByKeys) {
            queryParams.add("fieldsByKeys=true");
        }
        
        if (options.updateHistory) {
            queryParams.add("updateHistory=true");
        }
        
        if (!queryParams.isEmpty()) {
            url.append("?").append(String.join("&", queryParams));
        }
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url.toString()))
                .header("Authorization", authorization)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .GET()
                .build();
        
        HttpResponse<String> response = httpClient.send(request, 
                HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() != 200) {
            throw new RuntimeException("Failed to get issue: " + response.statusCode() + " - " + response.body());
        }
        
        return new JSONObject(response.body());
    }
    
    /**
     * 搜索Issues
     * @param searchOptions 搜索选项
     * @return JSONObject 搜索结果
     */
    public JSONObject searchIssues(IssueSearchOptions searchOptions) throws Exception {
        StringBuilder url = new StringBuilder(baseUrl + "rest/api/latest/search");
        
        List<String> queryParams = new ArrayList<>();
        
        if (searchOptions.jql != null && !searchOptions.jql.trim().isEmpty()) {
            queryParams.add("jql=" + URLEncoder.encode(searchOptions.jql, StandardCharsets.UTF_8));
        }
        
        queryParams.add("startAt=" + searchOptions.startAt);
        queryParams.add("maxResults=" + searchOptions.maxResults);
        
        if (searchOptions.fields != null && !searchOptions.fields.isEmpty()) {
            queryParams.add("fields=" + String.join(",", searchOptions.fields));
        }
        
        if (searchOptions.expand != null && !searchOptions.expand.isEmpty()) {
            queryParams.add("expand=" + String.join(",", searchOptions.expand));
        }
        
        if (searchOptions.properties != null && !searchOptions.properties.isEmpty()) {
            queryParams.add("properties=" + String.join(",", searchOptions.properties));
        }
        
        if (searchOptions.fieldsByKeys) {
            queryParams.add("fieldsByKeys=true");
        }
        
        url.append("?").append(String.join("&", queryParams));
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url.toString()))
                .header("Authorization", authorization)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .GET()
                .build();
        
        HttpResponse<String> response = httpClient.send(request, 
                HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() != 200) {
            throw new RuntimeException("Failed to search issues: " + response.statusCode() + " - " + response.body());
        }
        
        return new JSONObject(response.body());
    }
    
    /**
     * 创建Issue
     * @param issueData Issue数据
     * @return JSONObject 创建结果
     */
    public JSONObject createIssue(IssueData issueData) throws Exception {
        String url = baseUrl + "rest/api/latest/issue";
        
        JSONObject requestBody = buildIssueRequestBody(issueData);
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Authorization", authorization)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody.toString(), StandardCharsets.UTF_8))
                .build();
        
        HttpResponse<String> response = httpClient.send(request, 
                HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() != 201) {
            throw new RuntimeException("Failed to create issue: " + response.statusCode() + " - " + response.body());
        }
        
        return new JSONObject(response.body());
    }
    
    /**
     * 更新Issue
     * @param issueKeyOrId Issue Key或ID
     * @param issueData 更新数据
     * @param options 更新选项
     * @return 更新结果
     */
    public boolean updateIssue(String issueKeyOrId, IssueData issueData, IssueUpdateOptions options) throws Exception {
        StringBuilder url = new StringBuilder(baseUrl + "rest/api/latest/issue/" + issueKeyOrId);
        
        List<String> queryParams = new ArrayList<>();
        
        if (!options.notifyUsers) {
            queryParams.add("notifyUsers=false");
        }
        
        if (options.overrideScreenSecurity) {
            queryParams.add("overrideScreenSecurity=true");
        }
        
        if (options.overrideEditableFlag) {
            queryParams.add("overrideEditableFlag=true");
        }
        
        if (!queryParams.isEmpty()) {
            url.append("?").append(String.join("&", queryParams));
        }
        
        JSONObject requestBody = buildIssueRequestBody(issueData);
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url.toString()))
                .header("Authorization", authorization)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .PUT(HttpRequest.BodyPublishers.ofString(requestBody.toString(), StandardCharsets.UTF_8))
                .build();
        
        HttpResponse<String> response = httpClient.send(request, 
                HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() != 204) {
            throw new RuntimeException("Failed to update issue: " + response.statusCode() + " - " + response.body());
        }
        
        return true;
    }
    
    /**
     * 获取创建Issue的元数据
     * @param projectKey 项目Key
     * @param issueTypeName Issue类型名称
     * @return JSONObject 元数据信息
     */
    public JSONObject getCreateMetadata(String projectKey, String issueTypeName) throws Exception {
        StringBuilder url = new StringBuilder(baseUrl + "rest/api/latest/issue/createmeta");
        
        List<String> queryParams = new ArrayList<>();
        
        if (projectKey != null) {
            queryParams.add("projectKeys=" + projectKey);
        }
        
        if (issueTypeName != null) {
            queryParams.add("issuetypeNames=" + URLEncoder.encode(issueTypeName, StandardCharsets.UTF_8));
        }
        
        queryParams.add("expand=projects.issuetypes.fields");
        
        url.append("?").append(String.join("&", queryParams));
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url.toString()))
                .header("Authorization", authorization)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .GET()
                .build();
        
        HttpResponse<String> response = httpClient.send(request, 
                HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() != 200) {
            throw new RuntimeException("Failed to get create metadata: " + response.statusCode() + " - " + response.body());
        }
        
        return new JSONObject(response.body());
    }
    
    /**
     * 创建Task
     * @param projectKey 项目Key
     * @param summary 摘要
     * @param description 描述
     * @return JSONObject 创建结果
     */
    public JSONObject createTask(String projectKey, String summary, String description) throws Exception {
        IssueData taskData = new IssueData()
                .setProjectKey(projectKey)
                .setSummary(summary)
                .setDescription(description)
                .setIssueType(ISSUE_TYPE_TASK);
        
        return createIssue(taskData);
    }
    
    /**
     * 创建Story
     * @param projectKey 项目Key
     * @param summary 摘要
     * @param description 描述
     * @return JSONObject 创建结果
     */
    public JSONObject createStory(String projectKey, String summary, String description) throws Exception {
        IssueData storyData = new IssueData()
                .setProjectKey(projectKey)
                .setSummary(summary)
                .setDescription(description)
                .setIssueType(ISSUE_TYPE_STORY);
        
        return createIssue(storyData);
    }
    
    /**
     * 构建Issue请求体
     */
    private JSONObject buildIssueRequestBody(IssueData issueData) {
        JSONObject requestBody = new JSONObject();
        JSONObject fields = new JSONObject();
        
        // 基础字段
        if (issueData.projectKey != null) {
            fields.put(FIELD_PROJECT, new JSONObject().put("key", issueData.projectKey));
        }
        
        if (issueData.summary != null) {
            fields.put(FIELD_SUMMARY, issueData.summary);
        }
        
        if (issueData.description != null) {
            fields.put(FIELD_DESCRIPTION, issueData.description);
        }
        
        if (issueData.issueType != null) {
            fields.put(FIELD_ISSUE_TYPE, new JSONObject().put("name", issueData.issueType));
        }
        
        if (issueData.priority != null) {
            fields.put(FIELD_PRIORITY, new JSONObject().put("name", issueData.priority));
        }
        
        if (issueData.assigneeAccountId != null) {
            fields.put(FIELD_ASSIGNEE, new JSONObject().put("accountId", issueData.assigneeAccountId));
        }
        
        if (issueData.reporterAccountId != null) {
            fields.put(FIELD_REPORTER, new JSONObject().put("accountId", issueData.reporterAccountId));
        }
        
        if (issueData.labels != null && !issueData.labels.isEmpty()) {
            fields.put(FIELD_LABELS, new JSONArray(issueData.labels));
        }
        
        if (issueData.components != null && !issueData.components.isEmpty()) {
            JSONArray componentsArray = new JSONArray();
            for (String component : issueData.components) {
                componentsArray.put(new JSONObject().put("name", component));
            }
            fields.put(FIELD_COMPONENTS, componentsArray);
        }
        
        if (issueData.fixVersions != null && !issueData.fixVersions.isEmpty()) {
            JSONArray versionsArray = new JSONArray();
            for (String version : issueData.fixVersions) {
                versionsArray.put(new JSONObject().put("name", version));
            }
            fields.put(FIELD_FIX_VERSIONS, versionsArray);
        }
        
        if (issueData.versions != null && !issueData.versions.isEmpty()) {
            JSONArray versionsArray = new JSONArray();
            for (String version : issueData.versions) {
                versionsArray.put(new JSONObject().put("name", version));
            }
            fields.put(FIELD_VERSIONS, versionsArray);
        }
        
        if (issueData.dueDate != null) {
            fields.put(FIELD_DUE_DATE, issueData.dueDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd")));
        }
        
        if (issueData.environment != null) {
            fields.put(FIELD_ENVIRONMENT, issueData.environment);
        }
        
        // 自定义字段
        if (issueData.customFields != null) {
            for (Map.Entry<String, Object> entry : issueData.customFields.entrySet()) {
                fields.put(entry.getKey(), entry.getValue());
            }
        }
        
        requestBody.put("fields", fields);
        
        // 更新操作
        if (issueData.updateOperations != null && !issueData.updateOperations.isEmpty()) {
            requestBody.put("update", issueData.updateOperations);
        }
        
        // 历史元数据
        if (issueData.historyMetadata != null) {
            requestBody.put("historyMetadata", issueData.historyMetadata);
        }
        
        // 属性
        if (issueData.properties != null && !issueData.properties.isEmpty()) {
            JSONArray propertiesArray = new JSONArray();
            for (Map.Entry<String, Object> entry : issueData.properties.entrySet()) {
                JSONObject property = new JSONObject();
                property.put("key", entry.getKey());
                property.put("value", entry.getValue());
                propertiesArray.put(property);
            }
            requestBody.put("properties", propertiesArray);
        }
        
        return requestBody;
    }
    
    // 内部类定义
    
    /**
     * Issue查询选项
     */
    public static class IssueQueryOptions {
        public List<String> expand = new ArrayList<>();
        public List<String> fields = new ArrayList<>();
        public List<String> properties = new ArrayList<>();
        public boolean fieldsByKeys = false;
        public boolean updateHistory = false;
        
        public IssueQueryOptions expand(String... values) {
            this.expand.addAll(Arrays.asList(values));
            return this;
        }
        
        public IssueQueryOptions fields(String... values) {
            this.fields.addAll(Arrays.asList(values));
            return this;
        }
        
        public IssueQueryOptions properties(String... values) {
            this.properties.addAll(Arrays.asList(values));
            return this;
        }
        
        public IssueQueryOptions fieldsByKeys(boolean fieldsByKeys) {
            this.fieldsByKeys = fieldsByKeys;
            return this;
        }
        
        public IssueQueryOptions updateHistory(boolean updateHistory) {
            this.updateHistory = updateHistory;
            return this;
        }
    }
    
    /**
     * Issue搜索选项
     */
    public static class IssueSearchOptions {
        public String jql = "";
        public int startAt = 0;
        public int maxResults = 50;
        public List<String> fields = new ArrayList<>();
        public List<String> expand = new ArrayList<>();
        public List<String> properties = new ArrayList<>();
        public boolean fieldsByKeys = false;
        
        public IssueSearchOptions jql(String jql) {
            this.jql = jql;
            return this;
        }
        
        public IssueSearchOptions startAt(int startAt) {
            this.startAt = startAt;
            return this;
        }
        
        public IssueSearchOptions maxResults(int maxResults) {
            this.maxResults = Math.min(maxResults, 1000); // JIRA限制最大1000
            return this;
        }
        
        public IssueSearchOptions fields(String... values) {
            this.fields.addAll(Arrays.asList(values));
            return this;
        }
        
        public IssueSearchOptions expand(String... values) {
            this.expand.addAll(Arrays.asList(values));
            return this;
        }
        
        public IssueSearchOptions properties(String... values) {
            this.properties.addAll(Arrays.asList(values));
            return this;
        }
        
        public IssueSearchOptions fieldsByKeys(boolean fieldsByKeys) {
            this.fieldsByKeys = fieldsByKeys;
            return this;
        }
    }
    
    /**
     * Issue更新选项
     */
    public static class IssueUpdateOptions {
        public boolean notifyUsers = true;
        public boolean overrideScreenSecurity = false;
        public boolean overrideEditableFlag = false;
        
        public IssueUpdateOptions notifyUsers(boolean notifyUsers) {
            this.notifyUsers = notifyUsers;
            return this;
        }
        
        public IssueUpdateOptions overrideScreenSecurity(boolean overrideScreenSecurity) {
            this.overrideScreenSecurity = overrideScreenSecurity;
            return this;
        }
        
        public IssueUpdateOptions overrideEditableFlag(boolean overrideEditableFlag) {
            this.overrideEditableFlag = overrideEditableFlag;
            return this;
        }
    }
    
    /**
     * Issue数据类
     */
    public static class IssueData {
        // 基础字段
        public String projectKey;
        public String summary;
        public String description;
        public String issueType;
        public String priority;
        public String assigneeAccountId;
        public String reporterAccountId;
        public List<String> labels = new ArrayList<>();
        public List<String> components = new ArrayList<>();
        public List<String> fixVersions = new ArrayList<>();
        public List<String> versions = new ArrayList<>();
        public LocalDate dueDate;
        public String environment;
        
        // 扩展字段
        public Map<String, Object> customFields = new HashMap<>();
        public JSONObject updateOperations = new JSONObject();
        public JSONObject historyMetadata;
        public Map<String, Object> properties = new HashMap<>();
        
        // Setter方法（支持链式调用）
        public IssueData setProjectKey(String projectKey) {
            this.projectKey = projectKey;
            return this;
        }
        
        public IssueData setSummary(String summary) {
            this.summary = summary;
            return this;
        }
        
        public IssueData setDescription(String description) {
            this.description = description;
            return this;
        }
        
        public IssueData setIssueType(String issueType) {
            this.issueType = issueType;
            return this;
        }
        
        public IssueData setPriority(String priority) {
            this.priority = priority;
            return this;
        }
        
        public IssueData setAssigneeAccountId(String assigneeAccountId) {
            this.assigneeAccountId = assigneeAccountId;
            return this;
        }
        
        public IssueData setReporterAccountId(String reporterAccountId) {
            this.reporterAccountId = reporterAccountId;
            return this;
        }
        
        public IssueData addLabel(String label) {
            this.labels.add(label);
            return this;
        }
        
        public IssueData setLabels(List<String> labels) {
            this.labels = new ArrayList<>(labels);
            return this;
        }
        
        public IssueData addComponent(String component) {
            this.components.add(component);
            return this;
        }
        
        public IssueData setComponents(List<String> components) {
            this.components = new ArrayList<>(components);
            return this;
        }
        
        public IssueData addFixVersion(String version) {
            this.fixVersions.add(version);
            return this;
        }
        
        public IssueData setFixVersions(List<String> fixVersions) {
            this.fixVersions = new ArrayList<>(fixVersions);
            return this;
        }
        
        public IssueData addVersion(String version) {
            this.versions.add(version);
            return this;
        }
        
        public IssueData setVersions(List<String> versions) {
            this.versions = new ArrayList<>(versions);
            return this;
        }
        
        public IssueData setDueDate(LocalDate dueDate) {
            this.dueDate = dueDate;
            return this;
        }
        
        public IssueData setEnvironment(String environment) {
            this.environment = environment;
            return this;
        }
        
        public IssueData addCustomField(String fieldId, Object value) {
            this.customFields.put(fieldId, value);
            return this;
        }
        
        public IssueData setCustomFields(Map<String, Object> customFields) {
            this.customFields = new HashMap<>(customFields);
            return this;
        }
        
        public IssueData addProperty(String key, Object value) {
            this.properties.put(key, value);
            return this;
        }
        
        public IssueData setProperties(Map<String, Object> properties) {
            this.properties = new HashMap<>(properties);
            return this;
        }
        
        public IssueData setHistoryMetadata(JSONObject historyMetadata) {
            this.historyMetadata = historyMetadata;
            return this;
        }
        
        // 便捷方法：添加评论
        public IssueData addComment(String commentBody) {
            if (!updateOperations.has("comment")) {
                updateOperations.put("comment", new JSONArray());
            }
            JSONArray comments = updateOperations.getJSONArray("comment");
            JSONObject addComment = new JSONObject();
            addComment.put("add", new JSONObject().put("body", commentBody));
            comments.put(addComment);
            return this;
        }
        
        // 便捷方法：添加工作日志
        public IssueData addWorklog(String timeSpent, String comment, LocalDateTime started) {
            if (!updateOperations.has("worklog")) {
                updateOperations.put("worklog", new JSONArray());
            }
            JSONArray worklogs = updateOperations.getJSONArray("worklog");
            JSONObject addWorklog = new JSONObject();
            JSONObject worklogData = new JSONObject();
            worklogData.put("timeSpent", timeSpent);
            if (comment != null) {
                worklogData.put("comment", comment);
            }
            if (started != null) {
                worklogData.put("started", started.format(DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSZ")));
            }
            addWorklog.put("add", worklogData);
            worklogs.put(addWorklog);
            return this;
        }
        
        // Story Points设置（需要根据实际环境调整字段ID）
        public IssueData setStoryPoints(Integer storyPoints) {
            // 通常Story Points字段ID类似 customfield_10002，具体需要查询元数据获取
            this.customFields.put("customfield_10002", storyPoints);
            return this;
        }
        
        // Epic Link设置（需要根据实际环境调整字段ID）
        public IssueData setEpicLink(String epicKey) {
            // 通常Epic Link字段ID类似 customfield_10014，具体需要查询元数据获取
            this.customFields.put("customfield_10014", epicKey);
            return this;
        }
        
        // Sprint设置（需要根据实际环境调整字段ID）
        public IssueData setSprint(Integer sprintId) {
            // 通常Sprint字段ID类似 customfield_10020，具体需要查询元数据获取
            this.customFields.put("customfield_10020", sprintId);
            return this;
        }
    }
    
    /**
     * 主测试方法
     */
    public static void main(String[] args) {
        try {
            // 配置JIRA连接信息
            String jiraUrl = "https://wpb-jira.system.uk.hsbc/";
            String username = "45457904";
            String apiToken = "Sdw190310!";
            String projectKey = "AIWPB-78";
            
            // 创建JIRA客户端
            JiraApiClient jiraClient = new JiraApiClient(jiraUrl, username, apiToken);
            
            System.out.println("=== JIRA API 测试开始 ===");
            
            // 1. 获取创建元数据
            // System.out.println("\n1. 获取创建元数据...");
            // try {
            //     JSONObject metadata = jiraClient.getCreateMetadata(projectKey, ISSUE_TYPE_TASK);
            //     System.out.println("创建元数据获取成功");
            //     System.out.println("项目数量: " + metadata.getJSONArray("projects").length());
            // } catch (Exception e) {
            //     System.err.println("获取元数据失败: " + e.getMessage());
            // }
            
            // 2. 创建Task
            System.out.println("\n2. 创建Task...");
            try {
                IssueData taskData = new IssueData()
                        .setProjectKey(projectKey)
                        .setSummary("测试任务 - " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")))
                        .setDescription("这是通过API创建的测试任务\n\n包含以下特性：\n* 自动化测试\n* API集成\n* 完整字段支持")
                        .setIssueType(ISSUE_TYPE_TASK)
                        .setPriority("Medium")
                        .addLabel("api-test")
                        .addLabel("automation")
                        .setEnvironment("测试环境")
                        .setDueDate(LocalDate.now().plusDays(7))
                        .addProperty("source", "api-test");
                
                JSONObject createdTask = jiraClient.createIssue(taskData);
                String taskKey = createdTask.getString("key");
                System.out.println("Task创建成功，Key: " + taskKey);
                
                // 3. 创建Story
                System.out.println("\n3. 创建Story...");
                IssueData storyData = new IssueData()
                        .setProjectKey(projectKey)
                        .setSummary("用户故事 - API测试")
                        .setDescription("作为一个用户\n我希望能够通过API创建Story\n这样我就可以自动化管理需求")
                        .setIssueType(ISSUE_TYPE_STORY)
                        .setPriority("High")
                        .addLabel("user-story")
                        .addLabel("api-test")
                        .setStoryPoints(5)
                        .addComment("这是通过API创建的Story，包含完整的用户故事格式");
                
                JSONObject createdStory = jiraClient.createStory(projectKey, 
                        storyData.summary, storyData.description);
                String storyKey = createdStory.getString("key");
                System.out.println("Story创建成功，Key: " + storyKey);
                
                // 4. 查询刚创建的Task
                System.out.println("\n4. 查询Task详情...");
                IssueQueryOptions queryOptions = new IssueQueryOptions()
                        .fields("summary", "description", "status", "assignee", "reporter", "created")
                        .expand("changelog");
                
                JSONObject taskDetails = jiraClient.getIssue(taskKey, queryOptions);
                JSONObject taskFields = taskDetails.getJSONObject("fields");
                System.out.println("Task详情:");
                System.out.println("  摘要: " + taskFields.getString("summary"));
                System.out.println("  状态: " + taskFields.getJSONObject("status").getString("name"));
                System.out.println("  创建时间: " + taskFields.getString("created"));
                
                // 5. 更新Task
                System.out.println("\n5. 更新Task...");
                IssueData updateData = new IssueData()
                        .setSummary(taskFields.getString("summary") + " [已更新]")
                        .setDescription(taskFields.getString("description") + "\n\n=== 更新记录 ===\n更新时间: " + LocalDateTime.now())
                        .addLabel("updated")
                        .addComment("通过API更新了任务内容")
                        .addWorklog("30m", "进行了API测试工作", LocalDateTime.now());
                
                IssueUpdateOptions updateOptions = new IssueUpdateOptions()
                        .notifyUsers(false); // 不发送通知
                
                boolean updateResult = jiraClient.updateIssue(taskKey, updateData, updateOptions);
                System.out.println("Task更新" + (updateResult ? "成功" : "失败"));
                
                // 6. 搜索Issues
                System.out.println("\n6. 搜索Issues...");
                IssueSearchOptions searchOptions = new IssueSearchOptions()
                        .jql("project = " + projectKey + " AND labels = 'api-test' ORDER BY created DESC")
                        .maxResults(10)
                        .fields("key", "summary", "status", "created", "issuetype")
                        .expand("names");
                
                JSONObject searchResults = jiraClient.searchIssues(searchOptions);
                JSONArray issues = searchResults.getJSONArray("issues");
                
                System.out.println("搜索结果 (总计: " + searchResults.getInt("total") + " 条):");
                for (int i = 0; i < issues.length(); i++) {
                    JSONObject issue = issues.getJSONObject(i);
                    JSONObject fields = issue.getJSONObject("fields");
                    System.out.printf("  %s - %s [%s] (%s)%n", 
                            issue.getString("key"),
                            fields.getString("summary"),
                            fields.getJSONObject("status").getString("name"),
                            fields.getJSONObject("issuetype").getString("name"));
                }
                
                // 7. 查询特定JQL
                System.out.println("\n7. 高级搜索测试...");
                String advancedJql = String.format(
                        "project = %s AND created >= -1d AND (type = Story OR type = Task) ORDER BY priority DESC, created DESC",
                        projectKey);
                
                IssueSearchOptions advancedSearch = new IssueSearchOptions()
                        .jql(advancedJql)
                        .maxResults(5)
                        .fields("key", "summary", "priority", "assignee", "labels")
                        .fieldsByKeys(true);
                
                JSONObject advancedResults = jiraClient.searchIssues(advancedSearch);
                JSONArray advancedIssues = advancedResults.getJSONArray("issues");
                
                System.out.println("最近创建的Issues:");
                for (int i = 0; i < advancedIssues.length(); i++) {
                    JSONObject issue = advancedIssues.getJSONObject(i);
                    JSONObject fields = issue.getJSONObject("fields");
                    System.out.printf("  %s - %s [优先级: %s]%n", 
                            issue.getString("key"),
                            fields.getString("summary"),
                            fields.getJSONObject("priority").getString("name"));
                }
                
                System.out.println("\n=== JIRA API 测试完成 ===");
                
            } catch (Exception e) {
                System.err.println("创建或操作Issue失败: " + e.getMessage());
                e.printStackTrace();
            }
            
        } catch (Exception e) {
            System.err.println("JIRA API测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 辅助方法：格式化输出JSON（用于调试）
     */
    public static void printJson(JSONObject json) {
        System.out.println(json.toString(2));
    }
    
    /**
     * 辅助方法：创建带所有字段的完整Task示例
     */
    public static IssueData createFullTaskExample(String projectKey) {
        return new IssueData()
                .setProjectKey(projectKey)
                .setSummary("完整字段测试任务")
                .setDescription("这是一个包含所有可用字段的测试任务")
                .setIssueType(ISSUE_TYPE_TASK)
                .setPriority("High")
                .addLabel("complete-test")
                .addLabel("all-fields")
                .addComponent("API")
                .addComponent("Backend")
                .setEnvironment("开发环境\n操作系统: Linux\n数据库: MySQL")
                .setDueDate(LocalDate.now().plusDays(14))
                .addCustomField("customfield_10001", "自定义文本字段")
                .addCustomField("customfield_10003", Arrays.asList("选项1", "选项2"))
                .addProperty("priority_reason", "客户紧急需求")
                .addProperty("estimated_hours", 8)
                .addComment("这是通过API添加的初始评论")
                .addWorklog("2h", "需求分析", LocalDateTime.now().minusHours(2));
    }
    
    /**
     * 辅助方法：创建带所有字段的完整Story示例
     */
    public static IssueData createFullStoryExample(String projectKey) {
        return new IssueData()
                .setProjectKey(projectKey)
                .setSummary("完整用户故事示例")
                .setDescription("作为一个系统管理员\n我希望能够通过API管理所有Issue字段\n这样我就可以实现完全自动化的项目管理")
                .setIssueType(ISSUE_TYPE_STORY)
                .setPriority("Medium")
                .addLabel("user-story")
                .addLabel("automation")
                .addLabel("api-integration")
                .setStoryPoints(8)
                .setEnvironment("生产环境")
                .addCustomField("customfield_10005", "业务价值：提高工作效率")
                .addProperty("business_value", "high")
                .addProperty("technical_complexity", "medium")
                .addComment("验收标准：\n1. 支持所有标准字段\n2. 支持自定义字段\n3. 错误处理完善");
    }