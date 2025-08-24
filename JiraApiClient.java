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