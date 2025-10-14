import { Activity, TrendingUp, Zap, CheckCircle2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell } from "recharts";
import { useEffect, useState } from "react";
import { apiService, MetricsResponse, InteractionMetricsResponse, RewardsMetricsResponse, MetricsChartDataResponse } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface LearningData {
  time: string;
  successRate: number;
  reward: number;
  interactions: number;
}

interface RewardBreakdown {
  name: string;
  value: number;
  color: string;
}
export const MetricsOverview = () => {
  const [learningData, setLearningData] = useState<LearningData[]>([]);
  const [rewardBreakdown, setRewardBreakdown] = useState<RewardBreakdown[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [interactionMetrics, setInteractionMetrics] = useState<InteractionMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch metrics
        const metricsResponse = await apiService.getMetrics();
        setMetrics(metricsResponse);

        // Fetch interaction metrics
        const interactionResponse = await apiService.getInteractionMetrics();
        setInteractionMetrics(interactionResponse);

        // Fetch chart data
        const chartData = await apiService.getChartData();
        const learning: LearningData[] = (chartData as unknown as MetricsChartDataResponse).chart_data || [];
        setLearningData(learning);

        // Fetch reward breakdown
        const rewardResponse: RewardsMetricsResponse = await apiService.getRewardsMetrics();
        const rewards: RewardBreakdown[] = Object.entries(rewardResponse.current_metrics?.individual_rewards || {}).map(([key, value], index) => ({
          name: key.charAt(0).toUpperCase() + key.slice(1),
          value: value as number,
          color: `hsl(var(--chart-${index + 1}))`
        })) || [];
        setRewardBreakdown(rewards);

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch metrics data');
        console.error('Error fetching metrics data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
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
    );
  }

  if (error) {
    return (
      <Alert className="border-destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Failed to load metrics data: {error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Key Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="border-l-4 border-l-primary">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              {metrics && metrics.success_rate > 0 ? (metrics.success_rate * 100).toFixed(1) + '%' : 'No data'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {metrics && metrics.success_rate > 0 ? <span className="text-success">+5.3%</span> : ''} from last period
            </p>
            <Progress value={metrics && metrics.success_rate > 0 ? metrics.success_rate * 100 : 0} className="mt-3" />
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-accent">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Interactions</CardTitle>
            <Activity className="h-4 w-4 text-accent" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {interactionMetrics && interactionMetrics.total_interactions && interactionMetrics.total_interactions > 0 ? interactionMetrics.total_interactions.toLocaleString() : 'No data'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {interactionMetrics && interactionMetrics.total_interactions && interactionMetrics.total_interactions > 0 ? <span className="text-success">+1,234</span> : ''} today
            </p>
            <Progress value={interactionMetrics && interactionMetrics.total_interactions && interactionMetrics.total_interactions > 0 ? 75 : 0} className="mt-3" />
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-success">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Reward</CardTitle>
            <TrendingUp className="h-4 w-4 text-success" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {metrics && metrics.success_rate > 0 ? metrics.success_rate.toFixed(2) : 'No data'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {metrics && metrics.success_rate > 0 ? <span className="text-success">+0.12</span> : ''} improvement
            </p>
            <Progress value={metrics && metrics.success_rate > 0 ? metrics.success_rate * 100 : 0} className="mt-3" />
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-warning">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Learning Speed</CardTitle>
            <Zap className="h-4 w-4 text-warning" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{interactionMetrics && interactionMetrics.summary?.avg_interactions_per_hour && interactionMetrics.summary.avg_interactions_per_hour > 0 ? 'Fast' : 'No data'}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {interactionMetrics && interactionMetrics.summary?.avg_interactions_per_hour && interactionMetrics.summary.avg_interactions_per_hour > 0 ? `${interactionMetrics.summary.avg_interactions_per_hour} interactions/hour` : ''}
            </p>
            <Progress value={interactionMetrics && interactionMetrics.summary?.avg_interactions_per_hour && interactionMetrics.summary.avg_interactions_per_hour > 0 ? 85 : 0} className="mt-3" />
          </CardContent>
        </Card>
      </div>

      {/* Charts Row */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Learning Progress Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Learning Progress Over Time</CardTitle>
            <CardDescription>Success rate and reward trends (24h)</CardDescription>
          </CardHeader>
          <CardContent>
            {learningData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={learningData}>
                  <defs>
                    <linearGradient id="successGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--chart-1))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--chart-2))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--chart-2))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="time" />
                  <YAxis yAxisId="left" orientation="left" domain={[0, 100]} />
                  <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="successRate"
                    stroke="hsl(var(--chart-1))"
                    fill="url(#successGradient)"
                    name="Success Rate %"
                  />
                  <Area
                    yAxisId="right"
                    type="monotone"
                    dataKey="reward"
                    stroke="hsl(var(--chart-2))"
                    fill="url(#rewardGradient)"
                    name="Reward"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>No learning progress data available</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Reward Breakdown Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Multi-Objective Reward Breakdown</CardTitle>
            <CardDescription>Performance across different metrics</CardDescription>
          </CardHeader>
          <CardContent>
            {rewardBreakdown.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={rewardBreakdown} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis type="number" domain={[0, 1]} />
                  <YAxis dataKey="name" type="category" />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                    {rewardBreakdown.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>No reward breakdown data available</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Interactions Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Interaction Volume</CardTitle>
          <CardDescription>Total interactions processed over time</CardDescription>
        </CardHeader>
        <CardContent>
          {learningData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={learningData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="interactions"
                  stroke="hsl(var(--chart-3))"
                  strokeWidth={3}
                  dot={{ fill: "hsl(var(--chart-3))", r: 4 }}
                  name="Total Interactions"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>No interaction volume data available</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
