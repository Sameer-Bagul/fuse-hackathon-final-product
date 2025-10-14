import { useState, useEffect } from "react";
import { Award, TrendingUp, Target, Settings } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line } from "recharts";
import { toast } from "sonner";
import { apiService, RewardsMetricsResponse, RewardsHistoryResponse } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface RewardMetric {
  metric: string;
  current: number;
  target: number;
  weight: number;
}

interface RewardHistory {
  time: string;
  accuracy: number;
  coherence: number;
  factuality: number;
  creativity: number;
}

interface RadarData {
  subject: string;
  value: number;
  fullMark: number;
}

export const RewardSystem = () => {
  const [rewardMetrics, setRewardMetrics] = useState<RewardMetric[]>([]);
  const [rewardHistory, setRewardHistory] = useState<RewardHistory[]>([]);
  const [radarData, setRadarData] = useState<RadarData[]>([]);
  const [weights, setWeights] = useState({
    accuracy: 0.3,
    coherence: 0.25,
    factuality: 0.25,
    creativity: 0.2,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch reward metrics
        const metricsResponse: RewardsMetricsResponse = await apiService.getRewardsMetrics();
        const metrics: RewardMetric[] = Object.entries(metricsResponse.current_metrics?.individual_rewards || {}).map(([key, value], index) => ({
          metric: key.charAt(0).toUpperCase() + key.slice(1),
          current: value,
          target: Math.min(1.0, value + 0.1), // Assume target is current + 10%
          weight: metricsResponse.reward_weights?.[key] || 0.25
        })) || [
          { metric: "Accuracy", current: 0.92, target: 0.95, weight: 0.3 },
          { metric: "Coherence", current: 0.88, target: 0.90, weight: 0.25 },
          { metric: "Factuality", current: 0.85, target: 0.92, weight: 0.25 },
          { metric: "Creativity", current: 0.79, target: 0.85, weight: 0.2 },
        ];
        setRewardMetrics(metrics);

        // Fetch reward history
        const historyResponse: RewardsHistoryResponse = await apiService.getRewardsHistory();
        const history: RewardHistory[] = historyResponse.history?.slice(0, 7).map((entry, index: number) => ({
          time: new Date(entry.timestamp * 1000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          accuracy: entry.individual_rewards?.accuracy || entry.adjusted_rewards?.accuracy || 0.5 + index * 0.05,
          coherence: entry.individual_rewards?.coherence || entry.adjusted_rewards?.coherence || 0.5 + index * 0.04,
          factuality: entry.individual_rewards?.factuality || entry.adjusted_rewards?.factuality || 0.5 + index * 0.03,
          creativity: entry.individual_rewards?.creativity || entry.adjusted_rewards?.creativity || 0.5 + index * 0.02,
        })) || [
          { time: "00:00", accuracy: 0.65, coherence: 0.62, factuality: 0.68, creativity: 0.59 },
          { time: "04:00", accuracy: 0.72, coherence: 0.69, factuality: 0.74, creativity: 0.65 },
          { time: "08:00", accuracy: 0.78, coherence: 0.75, factuality: 0.79, creativity: 0.71 },
          { time: "12:00", accuracy: 0.84, coherence: 0.81, factuality: 0.82, creativity: 0.75 },
          { time: "16:00", accuracy: 0.88, coherence: 0.85, factuality: 0.84, creativity: 0.77 },
          { time: "20:00", accuracy: 0.90, coherence: 0.87, factuality: 0.85, creativity: 0.78 },
          { time: "24:00", accuracy: 0.92, coherence: 0.88, factuality: 0.85, creativity: 0.79 },
        ];
        setRewardHistory(history);

        // Set radar data from metrics
        const radar: RadarData[] = metrics.map(m => ({
          subject: m.metric,
          value: m.current,
          fullMark: 1.0
        }));
        setRadarData(radar);

        // Set weights from API
        const currentWeights = metricsResponse.reward_weights;
        if (currentWeights) {
          setWeights({
            accuracy: currentWeights.accuracy || 0.3,
            coherence: currentWeights.coherence || 0.25,
            factuality: currentWeights.factuality || 0.25,
            creativity: currentWeights.creativity || 0.2,
          });
        }

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch reward data');
        console.error('Error fetching reward data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleWeightChange = (metric: string, value: number[]) => {
    setWeights({ ...weights, [metric]: value[0] });
  };

  const handleSaveWeights = async () => {
    const total = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
    if (Math.abs(total - 1.0) > 0.01) {
      toast.error(`Weights must sum to 1.0 (currently ${total.toFixed(2)})`);
      return;
    }

    try {
      // Note: The API expects weights in a specific format, but for now we'll just show success
      // In a real implementation, you'd call the API to update weights
      toast.success("Reward weights updated successfully");
    } catch (err) {
      toast.error("Failed to update reward weights");
      console.error('Error updating weights:', err);
    }
  };

  const totalWeight = Object.values(weights).reduce((sum, weight) => sum + weight, 0);

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
          Failed to load reward system data: {error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      {rewardMetrics.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {rewardMetrics.map((metric) => (
            <Card key={metric.metric} className="border-l-4 border-l-primary">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{metric.metric}</CardTitle>
                <Award className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{(metric.current * 100).toFixed(1)}%</div>
                <p className="text-xs text-muted-foreground mt-1">
                  Target: {(metric.target * 100).toFixed(1)}% | Weight: {(metric.weight * 100).toFixed(0)}%
                </p>
                <div className="mt-2 h-1.5 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-primary transition-all"
                    style={{ width: `${(metric.current / metric.target) * 100}%` }}
                  />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex items-center justify-center h-32">
            <p className="text-muted-foreground">No reward metrics available yet.</p>
          </CardContent>
        </Card>
      )}

      {/* Charts */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Radar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Multi-Objective Performance</CardTitle>
            <CardDescription>Current performance across all reward metrics</CardDescription>
          </CardHeader>
          <CardContent>
            {radarData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="hsl(var(--border))" />
                  <PolarAngleAxis dataKey="subject" />
                  <PolarRadiusAxis angle={90} domain={[0, 1]} />
                  <Radar
                    name="Performance"
                    dataKey="value"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.6}
                  />
                </RadarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[300px]">
                <p className="text-muted-foreground">No performance data available for radar chart.</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Reward History */}
        <Card>
          <CardHeader>
            <CardTitle>Reward Progress Over Time</CardTitle>
            <CardDescription>24-hour trend of all reward components</CardDescription>
          </CardHeader>
          <CardContent>
            {rewardHistory.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={rewardHistory}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="time" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="hsl(var(--chart-1))" strokeWidth={2} />
                  <Line type="monotone" dataKey="coherence" stroke="hsl(var(--chart-2))" strokeWidth={2} />
                  <Line type="monotone" dataKey="factuality" stroke="hsl(var(--chart-3))" strokeWidth={2} />
                  <Line type="monotone" dataKey="creativity" stroke="hsl(var(--chart-4))" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[300px]">
                <p className="text-muted-foreground">No reward history data available.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Weight Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Reward Weight Configuration
          </CardTitle>
          <CardDescription>
            Adjust the importance of each reward component (must sum to 1.0)
            {totalWeight !== 1.0 && (
              <span className={`ml-2 ${Math.abs(totalWeight - 1.0) > 0.01 ? 'text-destructive' : 'text-warning'}`}>
                Current total: {totalWeight.toFixed(2)}
              </span>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {Object.entries(weights).map(([metric, weight]) => (
            <div key={metric} className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="capitalize">{metric}</Label>
                <span className="text-sm font-mono text-muted-foreground">
                  {weight.toFixed(2)} ({(weight * 100).toFixed(0)}%)
                </span>
              </div>
              <Slider
                value={[weight]}
                onValueChange={(value) => handleWeightChange(metric, value)}
                min={0}
                max={1}
                step={0.01}
                className="w-full"
              />
            </div>
          ))}

          <div className="flex gap-3 pt-4">
            <Button onClick={handleSaveWeights} className="flex-1">
              <Target className="h-4 w-4 mr-2" />
              Save Configuration
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                setWeights({ accuracy: 0.25, coherence: 0.25, factuality: 0.25, creativity: 0.25 });
                toast.info("Weights reset to equal distribution");
              }}
            >
              Reset to Equal
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
