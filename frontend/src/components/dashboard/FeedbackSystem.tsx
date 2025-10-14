import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from "recharts";
import {
  MessageSquare,
  TrendingUp,
  Users,
  Target,
  AlertCircle,
  CheckCircle,
  Send,
  BarChart3,
  PieChart as PieChartIcon,
  Activity
} from "lucide-react";
import {
  apiService,
  SubmitCorrectionRequest,
  SubmitCorrectionResponse,
  FeedbackAnalyticsResponse,
  SubmitErrorReportRequest,
  SubmitErrorReportResponse
} from "@/lib/api";

interface CorrectionFormData {
  userId: string;
  promptId: string;
  responseId: string;
  correctedResponse: string;
  correctionType: 'factual_error' | 'logical_error' | 'incomplete_info' | 'better_alternative' | 'style_improvement' | 'format_issue';
  explanation: string;
  improvementTags: string[];
}

interface ErrorReportFormData {
  userId: string;
  errorMessage: string;
  stackTrace: string;
  context: string;
  component: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

const CORRECTION_TYPES = [
  { value: 'factual_error', label: 'Factual Error', description: 'Incorrect facts or information' },
  { value: 'logical_error', label: 'Logical Error', description: 'Flawed reasoning or logic' },
  { value: 'incomplete_info', label: 'Incomplete Information', description: 'Missing important details' },
  { value: 'better_alternative', label: 'Better Alternative', description: 'A more effective approach' },
  { value: 'style_improvement', label: 'Style Improvement', description: 'Better phrasing or clarity' },
  { value: 'format_issue', label: 'Format Issue', description: 'Structure or presentation problems' }
];

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

export const FeedbackSystem = () => {
  const [activeTab, setActiveTab] = useState("submit");
  const [formData, setFormData] = useState<CorrectionFormData>({
    userId: "user_123", // Default for demo
    promptId: "",
    responseId: "",
    correctedResponse: "",
    correctionType: 'factual_error',
    explanation: "",
    improvementTags: []
  });

  const [analytics, setAnalytics] = useState<FeedbackAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [analyticsLoading, setAnalyticsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [tagInput, setTagInput] = useState("");

  const [errorReportForm, setErrorReportForm] = useState<ErrorReportFormData>({
    userId: "user_123", // Default for demo
    errorMessage: "",
    stackTrace: "",
    context: "",
    component: "",
    severity: 'medium'
  });

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      setAnalyticsLoading(true);
      setError(null);
      const data = await apiService.getFeedbackAnalytics();
      setAnalytics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch feedback analytics');
      console.error('Error fetching feedback analytics:', err);
    } finally {
      setAnalyticsLoading(false);
    }
  };

  const handleSubmitCorrection = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.correctedResponse.trim() || !formData.promptId.trim() || !formData.responseId.trim()) {
      setError("Please fill in all required fields");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      const request: SubmitCorrectionRequest = {
        user_id: formData.userId,
        prompt_id: formData.promptId,
        response_id: formData.responseId,
        corrected_response: formData.correctedResponse,
        correction_type: formData.correctionType,
        explanation: formData.explanation || undefined,
        improvement_tags: formData.improvementTags.length > 0 ? formData.improvementTags : undefined
      };

      const response: SubmitCorrectionResponse = await apiService.submitCorrection(request);

      if (response.success) {
        setSuccess(`Correction submitted successfully! ID: ${response.correction_id}`);
        // Reset form
        setFormData({
          ...formData,
          promptId: "",
          responseId: "",
          correctedResponse: "",
          explanation: "",
          improvementTags: []
        });
        setTagInput("");
        // Refresh analytics
        fetchAnalytics();
      } else {
        setError(response.message || "Failed to submit correction");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit correction');
      console.error('Error submitting correction:', err);
    } finally {
      setLoading(false);
    }
  };

  const addTag = () => {
    if (tagInput.trim() && !formData.improvementTags.includes(tagInput.trim())) {
      setFormData({
        ...formData,
        improvementTags: [...formData.improvementTags, tagInput.trim()]
      });
      setTagInput("");
    }
  };

  const removeTag = (tag: string) => {
    setFormData({
      ...formData,
      improvementTags: formData.improvementTags.filter(t => t !== tag)
    });
  };

  const handleSubmitErrorReport = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!errorReportForm.errorMessage.trim()) {
      setError("Please provide an error message");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      const request: SubmitErrorReportRequest = {
        user_id: errorReportForm.userId,
        error_message: errorReportForm.errorMessage,
        stack_trace: errorReportForm.stackTrace || undefined,
        context: errorReportForm.context || undefined,
        component: errorReportForm.component || undefined,
        severity: errorReportForm.severity,
        user_agent: navigator.userAgent,
        url: window.location.href
      };

      const response: SubmitErrorReportResponse = await apiService.submitErrorReport(request);

      if (response.success) {
        setSuccess(`Error report submitted successfully! ID: ${response.error_report_id}`);
        // Reset form
        setErrorReportForm({
          ...errorReportForm,
          errorMessage: "",
          stackTrace: "",
          context: "",
          component: ""
        });
      } else {
        setError(response.message || "Failed to submit error report");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit error report');
      console.error('Error submitting error report:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderAnalyticsCharts = () => {
    if (!analytics) return null;

    const { metrics } = analytics;

    // Prepare data for charts
    const categoryData = Object.entries(metrics.feedback_categories).map(([category, count]) => ({
      category: category.charAt(0).toUpperCase() + category.slice(1),
      count
    }));

    const correctionTypeData = Object.entries(metrics.correction_types).map(([type, count]) => ({
      type: type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
      count
    }));

    const adoptionData = [
      { name: 'Correction Adoption', value: metrics.correction_adoption_rate * 100, color: '#00C49F' },
      { name: 'Remaining', value: (1 - metrics.correction_adoption_rate) * 100, color: '#FF8042' }
    ];

    return (
      <div className="space-y-6">
        {/* Key Metrics Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Feedback</CardTitle>
              <MessageSquare className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metrics.total_feedbacks}</div>
              <p className="text-xs text-muted-foreground">
                {metrics.processed_feedbacks} processed
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Rating</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {metrics.average_rating ? metrics.average_rating.toFixed(1) : 'N/A'}
              </div>
              <p className="text-xs text-muted-foreground">Out of 5 stars</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Adoption Rate</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(metrics.correction_adoption_rate * 100).toFixed(1)}%
              </div>
              <Progress value={metrics.correction_adoption_rate * 100} className="mt-2" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Quality Score</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(metrics.feedback_quality_score * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground">Feedback quality</p>
            </CardContent>
          </Card>
        </div>

        {/* Charts */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Feedback Categories */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Feedback Categories
              </CardTitle>
              <CardDescription>Distribution of feedback by category</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={categoryData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="category" angle={-15} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="hsl(var(--primary))" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Correction Types */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChartIcon className="h-5 w-5" />
                Correction Types
              </CardTitle>
              <CardDescription>Types of corrections submitted</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={correctionTypeData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ type, percent }) => `${type}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="count"
                  >
                    {correctionTypeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Adoption Rate Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Correction Adoption Rate</CardTitle>
            <CardDescription>How often corrections are adopted by the system</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={adoptionData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(value) => [`${value}%`, 'Rate']} />
                <Bar dataKey="value" fill="#00C49F" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Feedback System</h2>
          <p className="text-muted-foreground">
            Submit corrections and view feedback analytics for continuous improvement
          </p>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList>
          <TabsTrigger value="submit">Submit Correction</TabsTrigger>
          <TabsTrigger value="error-report">Report Error</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="submit" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Submit Correction</CardTitle>
              <CardDescription>
                Provide corrections to improve AI responses and help the system learn
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmitCorrection} className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="promptId">Prompt ID *</Label>
                    <Input
                      id="promptId"
                      value={formData.promptId}
                      onChange={(e) => setFormData({ ...formData, promptId: e.target.value })}
                      placeholder="Enter the prompt ID"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="responseId">Response ID *</Label>
                    <Input
                      id="responseId"
                      value={formData.responseId}
                      onChange={(e) => setFormData({ ...formData, responseId: e.target.value })}
                      placeholder="Enter the response ID"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="correctionType">Correction Type *</Label>
                  <Select
                    value={formData.correctionType}
                    onValueChange={(value: string) => setFormData({ ...formData, correctionType: value as CorrectionFormData['correctionType'] })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select correction type" />
                    </SelectTrigger>
                    <SelectContent>
                      {CORRECTION_TYPES.map((type) => (
                        <SelectItem key={type.value} value={type.value}>
                          <div>
                            <div className="font-medium">{type.label}</div>
                            <div className="text-sm text-muted-foreground">{type.description}</div>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="correctedResponse">Corrected Response *</Label>
                  <Textarea
                    id="correctedResponse"
                    value={formData.correctedResponse}
                    onChange={(e) => setFormData({ ...formData, correctedResponse: e.target.value })}
                    placeholder="Enter the corrected version of the response"
                    rows={4}
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="explanation">Explanation (Optional)</Label>
                  <Textarea
                    id="explanation"
                    value={formData.explanation}
                    onChange={(e) => setFormData({ ...formData, explanation: e.target.value })}
                    placeholder="Explain why this correction improves the response"
                    rows={3}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Improvement Tags</Label>
                  <div className="flex gap-2">
                    <Input
                      value={tagInput}
                      onChange={(e) => setTagInput(e.target.value)}
                      placeholder="Add improvement tag"
                      onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                    />
                    <Button type="button" variant="outline" onClick={addTag}>
                      Add
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {formData.improvementTags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="cursor-pointer" onClick={() => removeTag(tag)}>
                        {tag} ×
                      </Badge>
                    ))}
                  </div>
                </div>

                {error && (
                  <Alert className="border-destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {success && (
                  <Alert className="border-success">
                    <CheckCircle className="h-4 w-4" />
                    <AlertDescription>{success}</AlertDescription>
                  </Alert>
                )}

                <Button type="submit" disabled={loading} className="w-full">
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Submitting...
                    </>
                  ) : (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      Submit Correction
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="error-report" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Report System Error</CardTitle>
              <CardDescription>
                Report bugs, errors, or issues encountered while using the system. Include as much detail as possible to help us fix the problem.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmitErrorReport} className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="errorSeverity">Severity *</Label>
                    <Select
                      value={errorReportForm.severity}
                      onValueChange={(value: string) => setErrorReportForm({ ...errorReportForm, severity: value as ErrorReportFormData['severity'] })}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select severity level" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="low">Low - Minor issue, doesn't affect functionality</SelectItem>
                        <SelectItem value="medium">Medium - Affects some functionality</SelectItem>
                        <SelectItem value="high">High - Major functionality broken</SelectItem>
                        <SelectItem value="critical">Critical - System unusable</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="errorComponent">Component (Optional)</Label>
                    <Input
                      id="errorComponent"
                      value={errorReportForm.component}
                      onChange={(e) => setErrorReportForm({ ...errorReportForm, component: e.target.value })}
                      placeholder="e.g., Dashboard, API, Learning System"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="errorMessage">Error Message *</Label>
                  <Textarea
                    id="errorMessage"
                    value={errorReportForm.errorMessage}
                    onChange={(e) => setErrorReportForm({ ...errorReportForm, errorMessage: e.target.value })}
                    placeholder="Describe the error or issue you encountered"
                    rows={3}
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="errorContext">Context (Optional)</Label>
                  <Textarea
                    id="errorContext"
                    value={errorReportForm.context}
                    onChange={(e) => setErrorReportForm({ ...errorReportForm, context: e.target.value })}
                    placeholder="What were you doing when the error occurred? Any additional context?"
                    rows={3}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="stackTrace">Stack Trace / Technical Details (Optional)</Label>
                  <Textarea
                    id="stackTrace"
                    value={errorReportForm.stackTrace}
                    onChange={(e) => setErrorReportForm({ ...errorReportForm, stackTrace: e.target.value })}
                    placeholder="Copy and paste any error messages, console logs, or technical details"
                    rows={4}
                    className="font-mono text-sm"
                  />
                </div>

                {error && (
                  <Alert className="border-destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {success && (
                  <Alert className="border-success">
                    <CheckCircle className="h-4 w-4" />
                    <AlertDescription>{success}</AlertDescription>
                  </Alert>
                )}

                <Button type="submit" disabled={loading} className="w-full">
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Submitting...
                    </>
                  ) : (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      Submit Error Report
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          {analyticsLoading ? (
            <div className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Card key={i}>
                    <CardHeader>
                      <Skeleton className="h-4 w-32" />
                    </CardHeader>
                    <CardContent>
                      <Skeleton className="h-8 w-16 mb-2" />
                      <Skeleton className="h-2 w-full" />
                    </CardContent>
                  </Card>
                ))}
              </div>
              <div className="grid gap-6 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <Skeleton className="h-6 w-48" />
                  </CardHeader>
                  <CardContent>
                    <Skeleton className="h-80 w-full" />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <Skeleton className="h-6 w-48" />
                  </CardHeader>
                  <CardContent>
                    <Skeleton className="h-80 w-full" />
                  </CardContent>
                </Card>
              </div>
            </div>
          ) : error ? (
            <Alert className="border-destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Failed to load analytics: {error}
              </AlertDescription>
            </Alert>
          ) : (
            renderAnalyticsCharts()
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};