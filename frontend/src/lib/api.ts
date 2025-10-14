import axios, { AxiosInstance, AxiosResponse } from 'axios';

// Create axios instance with base configuration
const api: AxiosInstance = axios.create({
  baseURL: 'http://localhost:8082',
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// API response types
export interface HealthResponse {
  status: string;
  timestamp: number;
  version: string;
  metrics: {
    total_interactions: number;
    success_rate: number;
    components: Record<string, boolean>;
  };
}

export interface MetricsResponse {
  success_rate: number;
  pattern_frequency: Record<string, number>;
  total_interactions: number;
}

export interface PromptRequest {
  prompt_text: string;
  use_external_llm?: boolean;
  provider?: string;
  model?: string;
}

export interface PromptResponse {
  response: string;
  action: number;
  metrics: Record<string, unknown>;
  external_llm_used?: boolean;
  external_llm_info?: Record<string, unknown>;
}

// Analytics interfaces
export interface Bottleneck {
  component: string;
  latency: number;
  impact: number;
}

export interface AnalyticsBottlenecksResponse {
  bottlenecks: Bottleneck[];
}

export interface AnalyticsInsightsResponse {
  performance_insights: string[];
  bottleneck_alerts: string[];
  optimization_opportunities: string[];
}

export interface ChartDataPoint {
  day: string;
  actual: number | null;
  predicted: number;
  confidence: number;
}

export interface ChartDataResponse {
  chart_data: ChartDataPoint[];
}

export interface MetricsChartDataResponse {
  chart_data: {
    time: string;
    successRate: number;
    reward: number;
    interactions: number;
  }[];
}

// Curriculum interfaces
export interface Skill {
  name: string;
  level: string;
  completed: boolean;
}

export interface CurriculumSkillsResponse {
  skills: Record<string, Skill[]>;
}

export interface CurriculumProgressResponse {
  progress: Record<string, number>;
}

export interface SkillGap {
  skill_name: string;
  gap_percentage: number;
  priority: string;
}

export interface CurriculumGapsResponse {
  gaps: SkillGap[];
}

export interface Recommendation {
  description?: string;
  title?: string;
  details?: string;
  explanation?: string;
  action?: string;
}

export interface CurriculumRecommendationsResponse {
  recommendations: Recommendation[];
}

// Curriculum learning session interfaces
export interface CurriculumStartRequest {
  user_prompt: string;
  num_tasks?: number;
}

export interface CurriculumStartResponse {
  curriculum_info: Record<string, unknown>;
  message: string;
}

export interface CurriculumTaskResponse {
  task_info: Record<string, unknown>;
  task_sequence_position: number;
  total_tasks: number;
}

export interface CurriculumProcessRequest {
  response: string;
  task_info: Record<string, unknown>;
  user_id?: string;
}

export interface CurriculumProcessResponse {
  result: Record<string, unknown>;
  curriculum_progress: Record<string, unknown>;
  next_task_available: boolean;
}

export interface CurriculumStatusResponse {
  active: boolean;
  current_task?: number;
  total_tasks?: number;
  completion_percentage?: number;
  remaining_tasks?: number;
  skills_covered?: string[];
  next_task?: Record<string, unknown>;
}

// Hallucination interfaces
export interface HallucinationMetricsResponse {
  metrics: {
    detection_rate: number;
    average_confidence: number;
    total_checks: number;
  };
}

export interface HallucinationTrend {
  timestamp: number;
  detection_rate: number;
  avg_confidence: number;
}

export interface HallucinationTrendsResponse {
  hourly_data: HallucinationTrend[];
}

export interface Indicator {
  type?: string;
  name?: string;
}

export interface HallucinationDetection {
  prompt?: string;
  text?: string;
  confidence: number;
  risk_level?: string;
  indicators?: Indicator[];
}

export interface HallucinationDetectionsResponse {
  detections: HallucinationDetection[];
}

export interface RiskLevel {
  count: number;
}

export interface HallucinationRiskDistributionResponse {
  risk_distribution: Record<string, RiskLevel>;
}

// Learning interfaces
export interface LearningProgressResponse {
  iteration_count: number;
  avg_iteration_time: number;
  estimated_time_remaining: string;
  total_episodes: number;
  current_reward: number;
  best_reward: number;
  improvement_rate: number;
  data_available?: boolean;
}

export interface LearningConfigResponse {
  llm_config: {
    alpha: number;
  };
}

export interface LearningHistoryItem {
  timestamp: number;
  prompt_text: string;
  action: number | null;
  reward: number | null;
  response: string;
  source: string;
}

export interface LearningHistoryResponse {
  interactions: LearningHistoryItem[];
  total_count: number;
}

// Autonomous learning interfaces
export interface AutonomousLearningStatusResponse {
  is_running: boolean;
  progress: Record<string, unknown>;
  timestamp: number;
}

// Metrics interfaces
export interface InteractionMetricsResponse {
  total_interactions: number;
  summary: {
    avg_interactions_per_hour: number;
  };
}

// Rewards interfaces
export interface RewardsMetricsResponse {
  current_metrics: {
    individual_rewards: Record<string, number>;
  };
  reward_weights: Record<string, number>;
}

export interface RewardsBreakdownResponse {
  current_breakdown: {
    individual_rewards: Record<string, number>;
  };
}

export interface RewardHistoryEntry {
  timestamp: number;
  individual_rewards?: Record<string, number>;
  adjusted_rewards?: Record<string, number>;
}

export interface RewardsHistoryResponse {
  history: RewardHistoryEntry[];
}

// Meta-learning interfaces
export interface MetaLearningStatusResponse {
  current_strategy: string;
  current_params: Record<string, unknown>;
  strategy_history: string[];
  active_rules: Record<string, unknown>[];
  performance_summary: Record<string, unknown>;
  recommendations: Record<string, unknown>[];
}

export interface MetaPerformanceMetricsResponse {
  current_strategy: string;
  current_params: Record<string, unknown>;
  strategy_performance: Record<string, Record<string, unknown>>;
  recent_performance: Record<string, unknown>[];
  adaptation_history: Record<string, unknown>[];
  context_awareness: Record<string, unknown>;
  strategy_effectiveness: Record<string, number>;
  learning_transfer_memory: Record<string, unknown>;
}

export interface MetaAdaptationTriggerResponse {
  success: boolean;
  message: string;
  new_strategy?: string;
  parameter_changes?: Record<string, unknown>;
}

export interface MetaStrategiesResponse {
  strategies: Record<string, unknown>[];
}

export interface MetaStrategySwitchRequest {
  strategy_name: string;
}

export interface MetaStrategySwitchResponse {
  success: boolean;
  message: string;
  old_strategy?: string;
  new_strategy?: string;
  available_strategies?: string[];
}

// Persistence interfaces
export type LearningStateType = "llm_model" | "meta_learning" | "history" | "curriculum" | "reward_system" | "analytics" | "feedback" | "complete_system";

export interface SaveStateRequest {
  state_type: LearningStateType;
  description?: string;
  include_related?: boolean;
}

export interface SaveStateResponse {
  success: boolean;
  version: string;
  state_type: LearningStateType;
  timestamp: string;
  message: string;
}

export interface LoadStateRequest {
  state_type: LearningStateType;
  version?: string;
  instance_id?: string;
}

export interface LoadStateResponse {
  success: boolean;
  state_type: LearningStateType;
  version: string;
  timestamp: string;
  data: Record<string, unknown>;
  message: string;
}

export interface VersionInfo {
  version: string;
  timestamp: string;
  description?: string;
  performance_metrics?: Record<string, number>;
  size_bytes?: number;
}

export interface ListVersionsRequest {
  state_type: LearningStateType;
  instance_id?: string;
  limit?: number;
}

export interface ListVersionsResponse {
  state_type: LearningStateType;
  versions: VersionInfo[];
  total_count: number;
  current_version?: string;
}

export interface RollbackRequest {
  state_type: LearningStateType;
  target_version: string;
  confirm_rollback?: boolean;
}

export interface RollbackResponse {
  success: boolean;
  state_type: LearningStateType;
  rolled_back_from: string;
  rolled_back_to: string;
  timestamp: string;
  message: string;
}

// Feedback interfaces
export interface SubmitFeedbackRequest {
  user_id: string;
  session_id?: string;
  prompt_id: string;
  response_id: string;
  feedback_type: 'rating' | 'correction' | 'preference' | 'comment' | 'thumbs_up' | 'thumbs_down';
  category: 'accuracy' | 'coherence' | 'factuality' | 'creativity' | 'usefulness' | 'relevance' | 'completeness' | 'clarity';
  rating?: number;
  comment?: string;
  correction_text?: string;
  correction_type?: 'factual_error' | 'logical_error' | 'incomplete_info' | 'better_alternative' | 'style_improvement' | 'format_issue';
  preference_data?: Record<string, unknown>;
}

export interface SubmitFeedbackResponse {
  feedback_id: string;
  success: boolean;
  message: string;
  processed_immediately: boolean;
}

export interface SubmitCorrectionRequest {
  user_id: string;
  prompt_id: string;
  response_id: string;
  corrected_response: string;
  correction_type: 'factual_error' | 'logical_error' | 'incomplete_info' | 'better_alternative' | 'style_improvement' | 'format_issue';
  explanation?: string;
  improvement_tags?: string[];
}

export interface SubmitCorrectionResponse {
  correction_id: string;
  success: boolean;
  message: string;
  validation_status: string;
}

export interface FeedbackAnalyticsResponse {
  metrics: {
    total_feedbacks: number;
    processed_feedbacks: number;
    average_rating?: number;
    feedback_categories: Record<string, number>;
    correction_types: Record<string, number>;
    user_engagement_rate: number;
    correction_adoption_rate: number;
    preference_learning_accuracy: number;
    collaborative_patterns_learned: number;
    feedback_quality_score: number;
  };
  timestamp: number;
  response_analysis?: {
    response_id: string;
    total_feedbacks: number;
    average_rating?: number;
    sentiment_score: number;
    common_issues: string[];
    improvement_suggestions: string[];
  };
  collaborative_insights: Record<string, unknown>[];
}

export interface SubmitErrorReportRequest {
  user_id: string;
  session_id?: string;
  error_message: string;
  stack_trace?: string;
  context?: string;
  component?: string;
  severity: string;
  user_agent?: string;
  url?: string;
}

export interface SubmitErrorReportResponse {
  error_report_id: string;
  success: boolean;
  message: string;
  processed_immediately: boolean;
}

// API functions
export const apiService = {
  // Health check endpoint
  async checkHealth(): Promise<HealthResponse> {
    const response: AxiosResponse<HealthResponse> = await api.get('/health');
    return response.data;
  },

  // Get learning metrics
  async getMetrics(): Promise<MetricsResponse> {
    const response: AxiosResponse<MetricsResponse> = await api.get('/prompt/metrics');
    return response.data;
  },

  // Handle prompt (legacy endpoint)
  async handlePrompt(request: PromptRequest): Promise<PromptResponse> {
    const response: AxiosResponse<PromptResponse> = await api.post('/prompt/handle', request);
    return response.data;
  },

  // Process user prompt (user-driven endpoint)
  async processUserPrompt(request: PromptRequest, userId?: string): Promise<PromptResponse> {
    const params = userId ? { user_id: userId } : {};
    const response: AxiosResponse<PromptResponse> = await api.post('/prompt/handle', request, { params });
    return response.data;
  },

  // Evaluate model
  async evaluate(request: { prompt_text?: string; num_episodes: number }) {
    const response = await api.post('/evaluate', request);
    return response.data;
  },

  // Schedule tasks
  async scheduleTasks(request: { num_tasks: number; prompt_text?: string }) {
    const response = await api.post('/schedule', request);
    return response.data;
  },

  // Get chart data
  async getChartData(): Promise<ChartDataResponse> {
    const response: AxiosResponse<ChartDataResponse> = await api.get('/visualize/chartjs');
    return response.data;
  },

  // Get learning progress
  async getLearningProgress() {
    const response = await api.get('/visualize/learning_progress');
    return response.data;
  },

  // Start learning loop
  async startLearning() {
    const response = await api.post('/learning/start');
    return response.data;
  },

  // Stop learning loop
  async stopLearning() {
    const response = await api.post('/learning/stop');
    return response.data;
  },

  // Get learning progress
  async getLearningLoopProgress(): Promise<LearningProgressResponse> {
    const response: AxiosResponse<LearningProgressResponse> = await api.get('/learning/progress');
    return response.data;
  },

  // Get learning config
  async getLearningConfig(): Promise<LearningConfigResponse> {
    const response: AxiosResponse<LearningConfigResponse> = await api.get('/learning/config');
    return response.data;
  },

  // Get learning history
  async getLearningHistory(limit: number = 100, offset: number = 0): Promise<LearningHistoryResponse> {
    const response: AxiosResponse<LearningHistoryResponse> = await api.get(`/learning/history?limit=${limit}&offset=${offset}`);
    return response.data;
  },

  // Autonomous learning endpoints
  async startAutonomousLearning(): Promise<{ success: boolean; message: string; status: string }> {
    const response = await api.post('/learning/autonomous/start');
    return response.data;
  },

  async stopAutonomousLearning(): Promise<{ success: boolean; message: string; status: string }> {
    const response = await api.post('/learning/autonomous/stop');
    return response.data;
  },

  async getAutonomousLearningStatus(): Promise<AutonomousLearningStatusResponse> {
    const response: AxiosResponse<AutonomousLearningStatusResponse> = await api.get('/learning/autonomous/status');
    return response.data;
  },

  // Analytics endpoints
  async getAnalyticsBottlenecks(learnerId?: string): Promise<AnalyticsBottlenecksResponse> {
    const params = learnerId ? { learner_id: learnerId } : {};
    const response: AxiosResponse<AnalyticsBottlenecksResponse> = await api.get('/analytics/bottlenecks', { params });
    return response.data;
  },

  async getAnalyticsInsights(learnerId?: string): Promise<AnalyticsInsightsResponse> {
    const params = learnerId ? { learner_id: learnerId } : {};
    const response: AxiosResponse<AnalyticsInsightsResponse> = await api.get('/analytics/insights', { params });
    return response.data;
  },

  // Curriculum endpoints
  async getCurriculumProgress(learnerId: string): Promise<CurriculumProgressResponse> {
    const response: AxiosResponse<CurriculumProgressResponse> = await api.get(`/curriculum/progress?learner_id=${learnerId}`);
    return response.data;
  },

  async getCurriculumSkills(): Promise<CurriculumSkillsResponse> {
    const response: AxiosResponse<CurriculumSkillsResponse> = await api.get('/curriculum/skills');
    return response.data;
  },

  async getCurriculumGaps(learnerId: string): Promise<CurriculumGapsResponse> {
    const response: AxiosResponse<CurriculumGapsResponse> = await api.get(`/curriculum/gaps?learner_id=${learnerId}`);
    return response.data;
  },

  async getCurriculumRecommendations(learnerId: string): Promise<CurriculumRecommendationsResponse> {
    const response: AxiosResponse<CurriculumRecommendationsResponse> = await api.get(`/curriculum/recommendations?learner_id=${learnerId}`);
    return response.data;
  },

  // Curriculum learning session management
  async startCurriculumLearning(userPrompt: string, numTasks: number = 5): Promise<CurriculumStartResponse> {
    const response: AxiosResponse<CurriculumStartResponse> = await api.post('/curriculum/start', { user_prompt: userPrompt, num_tasks: numTasks });
    return response.data;
  },

  async getNextCurriculumTask(): Promise<CurriculumTaskResponse> {
    const response: AxiosResponse<CurriculumTaskResponse> = await api.get('/curriculum/task');
    return response.data;
  },

  async processCurriculumResponse(response: string, taskInfo: Record<string, unknown>, userId?: string): Promise<CurriculumProcessResponse> {
    const requestData: CurriculumProcessRequest = { response, task_info: taskInfo };
    if (userId) requestData.user_id = userId;
    const apiResponse: AxiosResponse<CurriculumProcessResponse> = await api.post('/curriculum/process', requestData);
    return apiResponse.data;
  },

  async getCurriculumStatus(): Promise<CurriculumStatusResponse> {
    const response: AxiosResponse<CurriculumStatusResponse> = await api.get('/curriculum/status');
    return response.data;
  },

  async stopCurriculumLearning(): Promise<{ message: string }> {
    const response = await api.post('/curriculum/stop');
    return response.data;
  },

  // Hallucination endpoints
  async getHallucinationMetrics(): Promise<HallucinationMetricsResponse> {
    const response: AxiosResponse<HallucinationMetricsResponse> = await api.get('/hallucination/metrics');
    return response.data;
  },

  async getHallucinationTrends(hours: number = 24): Promise<HallucinationTrendsResponse> {
    const response: AxiosResponse<HallucinationTrendsResponse> = await api.get(`/hallucination/trends?hours=${hours}`);
    return response.data;
  },

  async getHallucinationDetections(limit: number = 50): Promise<HallucinationDetectionsResponse> {
    const response: AxiosResponse<HallucinationDetectionsResponse> = await api.get(`/hallucination/detections?limit=${limit}`);
    return response.data;
  },

  async getHallucinationRiskDistribution(hours: number = 24): Promise<HallucinationRiskDistributionResponse> {
    const response: AxiosResponse<HallucinationRiskDistributionResponse> = await api.get(`/hallucination/risk-distribution?hours=${hours}`);
    return response.data;
  },

  // Metrics endpoints
  async getInteractionMetrics(hours: number = 24): Promise<InteractionMetricsResponse> {
    const response: AxiosResponse<InteractionMetricsResponse> = await api.get(`/metrics/interactions?hours=${hours}`);
    return response.data;
  },

  // Rewards endpoints
  async getRewardsMetrics(): Promise<RewardsMetricsResponse> {
    const response: AxiosResponse<RewardsMetricsResponse> = await api.get('/rewards/metrics');
    return response.data;
  },

  async getRewardsBreakdown(): Promise<RewardsBreakdownResponse> {
    const response: AxiosResponse<RewardsBreakdownResponse> = await api.get('/rewards/breakdown');
    return response.data;
  },

  async getRewardsHistory(limit: number = 100): Promise<RewardsHistoryResponse> {
    const response: AxiosResponse<RewardsHistoryResponse> = await api.get(`/rewards/history?limit=${limit}`);
    return response.data;
  },

  async getRewardsWeights() {
    const response = await api.get('/rewards/weights');
    return response.data;
  },

  // Feedback endpoints
  async submitFeedback(request: SubmitFeedbackRequest): Promise<SubmitFeedbackResponse> {
    const response: AxiosResponse<SubmitFeedbackResponse> = await api.post('/feedback/submit', request);
    return response.data;
  },

  async submitCorrection(request: SubmitCorrectionRequest): Promise<SubmitCorrectionResponse> {
    const response: AxiosResponse<SubmitCorrectionResponse> = await api.post('/feedback/correct', request);
    return response.data;
  },

  async getFeedbackAnalytics(userId?: string, responseId?: string): Promise<FeedbackAnalyticsResponse> {
    const params: Record<string, string> = {};
    if (userId) params.user_id = userId;
    if (responseId) params.response_id = responseId;
    const response: AxiosResponse<FeedbackAnalyticsResponse> = await api.get('/feedback/analytics', { params });
    return response.data;
  },

  async submitErrorReport(request: SubmitErrorReportRequest): Promise<SubmitErrorReportResponse> {
    const response: AxiosResponse<SubmitErrorReportResponse> = await api.post('/feedback/error-report', request);
    return response.data;
  },

  // Meta-learning endpoints
  async getMetaLearningStatus(): Promise<MetaLearningStatusResponse> {
    const response: AxiosResponse<MetaLearningStatusResponse> = await api.get('/meta/status');
    return response.data;
  },

  async getMetaPerformanceMetrics(): Promise<MetaPerformanceMetricsResponse> {
    const response: AxiosResponse<MetaPerformanceMetricsResponse> = await api.get('/meta/performance');
    return response.data;
  },

  async triggerMetaAdaptation(): Promise<MetaAdaptationTriggerResponse> {
    const response: AxiosResponse<MetaAdaptationTriggerResponse> = await api.post('/meta/adapt');
    return response.data;
  },

  async getAvailableStrategies(): Promise<MetaStrategiesResponse> {
    const response: AxiosResponse<MetaStrategiesResponse> = await api.get('/meta/strategies');
    return response.data;
  },

  async switchLearningStrategy(request: MetaStrategySwitchRequest): Promise<MetaStrategySwitchResponse> {
    const response: AxiosResponse<MetaStrategySwitchResponse> = await api.post('/meta/strategy', request);
    return response.data;
  },

  // Persistence endpoints
  async saveLearningState(request: SaveStateRequest): Promise<SaveStateResponse> {
    const response: AxiosResponse<SaveStateResponse> = await api.post('/persistence/save', request);
    return response.data;
  },

  async loadLearningState(request: LoadStateRequest): Promise<LoadStateResponse> {
    const response: AxiosResponse<LoadStateResponse> = await api.post('/persistence/load', request);
    return response.data;
  },

  async listStateVersions(request: ListVersionsRequest): Promise<ListVersionsResponse> {
    const params = new URLSearchParams();
    params.append('state_type', request.state_type as string);
    if (request.instance_id) params.append('instance_id', request.instance_id);
    if (request.limit !== undefined) params.append('limit', request.limit.toString());

    const response: AxiosResponse<ListVersionsResponse> = await api.get(`/persistence/versions?${params.toString()}`);
    return response.data;
  },

  async rollbackState(request: RollbackRequest): Promise<RollbackResponse> {
    const response: AxiosResponse<RollbackResponse> = await api.post('/persistence/rollback', request);
    return response.data;
  },

  async resetAllLearningData(confirmReset: boolean = false) {
    const params = new URLSearchParams();
    params.append('confirm_reset', confirmReset.toString());

    const response = await api.post(`/persistence/reset?${params.toString()}`);
    return response.data;
  },
};

// Export the axios instance for direct use if needed
export default api;