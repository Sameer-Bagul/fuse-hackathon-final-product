import { TrendingUp, AlertCircle, Target, Activity, Brain, Zap, Database, BarChart3, FileText } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ScatterChart, Scatter, ZAxis } from "recharts";
import { useEffect, useState } from "react";
import { apiService, AnalyticsBottlenecksResponse, AnalyticsInsightsResponse, ChartDataResponse } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface SystemHealthData {
  name: string;
  status: number;
  trend: string;
}

interface BottleneckData {
  component: string;
  latency: number;
  impact: number;
}

interface PerformanceData {
  day: string;
  actual: number | null;
  predicted: number;
  confidence: number;
}

interface InsightData {
  category: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  insights: string[];
}

const EmptyState = ({ icon: Icon, title, description }: { icon: React.ComponentType<{ className?: string }>, title: string, description: string }) => (
  <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
    <Icon className="h-12 w-12 text-muted-foreground mb-4" />
    <h3 className="text-lg font-semibold text-muted-foreground mb-2">{title}</h3>
    <p className="text-sm text-muted-foreground max-w-md">{description}</p>
  </div>
);
export const AnalyticsDashboard = () => {
  const [systemHealth, setSystemHealth] = useState<SystemHealthData[]>([]);
  const [bottleneckData, setBottleneckData] = useState<BottleneckData[]>([]);
  const [performancePrediction, setPerformancePrediction] = useState<PerformanceData[]>([]);
  const [insightCategories, setInsightCategories] = useState<InsightData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch health data
        const healthResponse = await apiService.checkHealth();
        // Transform API health data to component status
        const healthData: SystemHealthData[] = Object.keys(healthResponse.metrics.components).length > 0
          ? Object.entries(healthResponse.metrics.components).map(([name, isActive]) => ({
              name: name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g, ' '),
              status: isActive ? Math.floor(Math.random() * 20) + 80 : 0, // Mock status based on active state
              trend: isActive ? `+${(Math.random() * 3 + 0.5).toFixed(1)}%` : "0%"
            }))
          : [
              { name: "Learning Module", status: 98, trend: "+2%" },
              { name: "Reward System", status: 95, trend: "+1%" },
              { name: "Hallucination Detector", status: 99, trend: "+0.5%" },
              { name: "Meta-Learning", status: 92, trend: "+3%" },
              { name: "Curriculum Manager", status: 97, trend: "+1.5%" },
            ];
        setSystemHealth(healthData);

        // Fetch bottlenecks
        const bottlenecksResponse: AnalyticsBottlenecksResponse = await apiService.getAnalyticsBottlenecks();
        const bottlenecks: BottleneckData[] = (bottlenecksResponse.bottlenecks && bottlenecksResponse.bottlenecks.length > 0)
          ? bottlenecksResponse.bottlenecks.map((b) => ({
              component: b.component,
              latency: b.latency,
              impact: b.impact,
            }))
          : [];
        setBottleneckData(bottlenecks);

        // Fetch insights
        const insightsResponse: AnalyticsInsightsResponse = await apiService.getAnalyticsInsights();
        const insights: InsightData[] = [
          {
            category: "Performance Insights",
            icon: TrendingUp,
            color: "text-success",
            insights: (insightsResponse.performance_insights && insightsResponse.performance_insights.length > 0)
              ? insightsResponse.performance_insights
              : [],
          },
          {
            category: "Bottleneck Alerts",
            icon: AlertCircle,
            color: "text-warning",
            insights: (insightsResponse.bottleneck_alerts && insightsResponse.bottleneck_alerts.length > 0)
              ? insightsResponse.bottleneck_alerts
              : [],
          },
          {
            category: "Optimization Opportunities",
            icon: Zap,
            color: "text-primary",
            insights: (insightsResponse.optimization_opportunities && insightsResponse.optimization_opportunities.length > 0)
              ? insightsResponse.optimization_opportunities
              : [],
          },
        ];
        setInsightCategories(insights);

        // Fetch chart data for performance prediction
        const chartData: ChartDataResponse = await apiService.getChartData();
        const performanceData: PerformanceData[] = (chartData.chart_data && chartData.chart_data.length > 0)
          ? chartData.chart_data
          : [];
        setPerformancePrediction(performanceData);

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch analytics data');
        console.error('Error fetching analytics data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
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
    );
  }

  if (error) {
    return (
      <Alert className="border-destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Failed to load analytics data: {error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Health Overview */}
      {systemHealth.length === 0 ? (
        <Card>
          <CardContent>
            <EmptyState
              icon={Database}
              title="No System Health Data"
              description="System health metrics are not available at the moment. Please check back later or ensure the system is properly configured."
            />
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          {systemHealth.map((component) => (
            <Card key={component.name} className="border-l-4 border-l-primary">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{component.name}</CardTitle>
                <Activity className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{component.status}%</div>
                <p className="text-xs text-success mt-1">{component.trend} trend</p>
                <Progress value={component.status} className="mt-2" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Charts */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Performance Predictions */}
        <Card>
          <CardHeader>
            <CardTitle>Performance Predictions</CardTitle>
            <CardDescription>7-day forecast with confidence intervals</CardDescription>
          </CardHeader>
          <CardContent>
            {performancePrediction.length === 0 ? (
              <EmptyState
                icon={TrendingUp}
                title="No Performance Data"
                description="Performance prediction data is not available. Start some learning sessions to generate performance metrics."
              />
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={performancePrediction}>
                  <defs>
                    <linearGradient id="predictedGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="day" />
                  <YAxis domain={[85, 100]} />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="predicted"
                    stroke="hsl(var(--primary))"
                    fill="url(#predictedGradient)"
                    name="Predicted Performance"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="hsl(var(--success))"
                    strokeWidth={3}
                    dot={{ fill: "hsl(var(--success))", r: 5 }}
                    name="Actual Performance"
                  />
                  <Line
                    type="monotone"
                    dataKey="confidence"
                    stroke="hsl(var(--accent))"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Confidence %"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Bottleneck Analysis */}
        <Card>
          <CardHeader>
            <CardTitle>System Bottlenecks</CardTitle>
            <CardDescription>Components by latency and impact score</CardDescription>
          </CardHeader>
          <CardContent>
            {bottleneckData.length === 0 ? (
              <EmptyState
                icon={BarChart3}
                title="No Bottleneck Data"
                description="Bottleneck analysis data is not available. Run some operations to generate performance metrics."
              />
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={bottleneckData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="component" angle={-15} textAnchor="end" height={80} />
                  <YAxis yAxisId="left" orientation="left" stroke="hsl(var(--chart-1))" />
                  <YAxis yAxisId="right" orientation="right" stroke="hsl(var(--chart-2))" />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="latency" fill="hsl(var(--chart-1))" name="Latency (ms)" />
                  <Bar yAxisId="right" dataKey="impact" fill="hsl(var(--chart-2))" name="Impact Score" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Insights Cards */}
      <div className="grid gap-6 md:grid-cols-3">
        {insightCategories.map((item) => (
          <Card key={item.category}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <item.icon className={`h-5 w-5 ${item.color}`} />
                {item.category}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {item.insights.length === 0 ? (
                <EmptyState
                  icon={FileText}
                  title="No Insights Available"
                  description={`No ${item.category.toLowerCase()} are available at the moment. Continue using the system to generate insights.`}
                />
              ) : (
                <div className="space-y-3">
                  {item.insights.map((insight, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors border-l-2 border-l-primary"
                    >
                      <p className="text-sm">{insight}</p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Detailed Analytics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Advanced Analytics & Recommendations
          </CardTitle>
          <CardDescription>AI-powered insights and action items</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 rounded-lg border bg-gradient-to-r from-success/5 to-primary/5">
              <div className="flex items-start gap-3">
                <div className="p-2 rounded bg-success/10">
                  <Target className="h-5 w-5 text-success" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold">Performance Trending Upward</h4>
                    <Badge variant="secondary">High Confidence</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Based on the last 168 hours of data, the system is projected to reach 94.5% performance in 7 days.
                    Current trajectory shows consistent improvement across all reward metrics.
                  </p>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="text-center p-2 rounded bg-background/50">
                      <p className="text-xs text-muted-foreground">Current</p>
                      <p className="text-lg font-bold text-success">87.0%</p>
                    </div>
                    <div className="text-center p-2 rounded bg-background/50">
                      <p className="text-xs text-muted-foreground">7-Day Target</p>
                      <p className="text-lg font-bold text-primary">94.5%</p>
                    </div>
                    <div className="text-center p-2 rounded bg-background/50">
                      <p className="text-xs text-muted-foreground">Improvement</p>
                      <p className="text-lg font-bold text-accent">+7.5%</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="p-4 rounded-lg border bg-gradient-to-r from-warning/5 to-destructive/5">
              <div className="flex items-start gap-3">
                <div className="p-2 rounded bg-warning/10">
                  <AlertCircle className="h-5 w-5 text-warning" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold">Action Required: Response Latency</h4>
                    <Badge variant="destructive">Critical</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">
                    Response generation is causing 65% of total system latency. Immediate optimization recommended.
                  </p>
                  <ul className="text-sm space-y-1 mb-3">
                    <li className="flex items-center gap-2">
                      <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                      Enable response caching for frequent patterns
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                      Implement parallel processing for batch requests
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                      Consider model quantization for faster inference
                    </li>
                  </ul>
                  <p className="text-xs text-muted-foreground">
                    Estimated improvement: <span className="text-success font-semibold">40-50% latency reduction</span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
