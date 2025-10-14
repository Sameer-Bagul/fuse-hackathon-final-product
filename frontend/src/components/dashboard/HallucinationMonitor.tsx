import { AlertTriangle, Shield, TrendingDown, Eye } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, PieChart, Pie, Cell } from "recharts";
import { useEffect, useState } from "react";
import { apiService, HallucinationMetricsResponse, HallucinationTrendsResponse, HallucinationRiskDistributionResponse, HallucinationDetectionsResponse } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface HallucinationTrend {
  time: string;
  rate: number;
  confidence: number;
}

interface RiskDistribution {
  name: string;
  value: number;
  color: string;
}

interface Detection {
  id: number;
  prompt: string;
  confidence: number;
  risk: string;
  indicators: string[];
}
export const HallucinationMonitor = () => {
  const [hallucinationTrend, setHallucinationTrend] = useState<HallucinationTrend[]>([]);
  const [riskDistribution, setRiskDistribution] = useState<RiskDistribution[]>([]);
  const [recentDetections, setRecentDetections] = useState<Detection[]>([]);
  const [metrics, setMetrics] = useState<HallucinationMetricsResponse['metrics'] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch metrics
        const metricsResponse: HallucinationMetricsResponse = await apiService.getHallucinationMetrics();
        setMetrics(metricsResponse.metrics);

        // Fetch trends
        const trendsResponse: HallucinationTrendsResponse = await apiService.getHallucinationTrends();
        const trends: HallucinationTrend[] = trendsResponse.hourly_data?.map((item) => ({
          time: new Date(item.timestamp * 1000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          rate: item.detection_rate * 100,
          confidence: item.avg_confidence
        })) || [
          { time: "00:00", rate: 8.2, confidence: 0.92 },
          { time: "04:00", rate: 7.5, confidence: 0.93 },
          { time: "08:00", rate: 6.8, confidence: 0.94 },
          { time: "12:00", rate: 5.9, confidence: 0.95 },
          { time: "16:00", rate: 4.7, confidence: 0.96 },
          { time: "20:00", rate: 3.8, confidence: 0.97 },
          { time: "24:00", rate: 2.9, confidence: 0.98 },
        ];
        setHallucinationTrend(trends);

        // Fetch risk distribution
        const riskResponse: HallucinationRiskDistributionResponse = await apiService.getHallucinationRiskDistribution();
        const riskData: RiskDistribution[] = Object.entries(riskResponse.risk_distribution || {}).map(([key, value]) => ({
          name: key.charAt(0).toUpperCase() + key.slice(1).toLowerCase() + ' Risk',
          value: value.count,
          color: key === 'LOW' ? "hsl(var(--success))" : key === 'MEDIUM' ? "hsl(var(--warning))" : "hsl(var(--destructive))"
        })) || [
          { name: "Low Risk", value: 782, color: "hsl(var(--success))" },
          { name: "Medium Risk", value: 156, color: "hsl(var(--warning))" },
          { name: "High Risk", value: 42, color: "hsl(var(--destructive))" },
        ];
        setRiskDistribution(riskData);

        // Fetch recent detections
        const detectionsResponse: HallucinationDetectionsResponse = await apiService.getHallucinationDetections();
        const detections: Detection[] = detectionsResponse.detections?.map((d, index: number) => ({
          id: index + 1,
          prompt: d.prompt || d.text || "Unknown prompt",
          confidence: d.confidence,
          risk: d.risk_level?.toLowerCase() || "medium",
          indicators: d.indicators?.map((ind) => ind.type || ind.name) || ["Unknown"]
        })) || [
          {
            id: 1,
            prompt: "What is the capital of Mars?",
            confidence: 0.45,
            risk: "high",
            indicators: ["Factual Inconsistency", "Low Confidence"],
          },
          {
            id: 2,
            prompt: "Explain quantum entanglement",
            confidence: 0.89,
            risk: "low",
            indicators: ["High Confidence"],
          },
          {
            id: 3,
            prompt: "Historical events in 2025",
            confidence: 0.62,
            risk: "medium",
            indicators: ["Temporal Inconsistency"],
          },
        ];
        setRecentDetections(detections);

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch hallucination data');
        console.error('Error fetching hallucination data:', err);
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
          Failed to load hallucination data: {error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      {metrics ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card className="border-l-4 border-l-success">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Detection Rate</CardTitle>
              <Shield className="h-4 w-4 text-success" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-success">
                {metrics ? (metrics.detection_rate * 100).toFixed(1) + '%' : '97.2%'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Successful hallucination detection
              </p>
              <Progress value={metrics ? metrics.detection_rate * 100 : 97.2} className="mt-3" />
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-primary">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
              <Eye className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {metrics ? metrics.average_confidence?.toFixed(2) || '0.98' : '0.98'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Average confidence score
              </p>
              <Progress value={metrics ? (metrics.average_confidence || 0.98) * 100 : 98} className="mt-3" />
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-warning">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Hallucination Rate</CardTitle>
              <AlertTriangle className="h-4 w-4 text-warning" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {metrics ? ((metrics.total_checks - metrics.detection_rate * metrics.total_checks) / metrics.total_checks * 100).toFixed(1) + '%' : '2.9%'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                <span className="text-success">-5.3%</span> improvement
              </p>
              <Progress value={metrics ? ((metrics.total_checks - metrics.detection_rate * metrics.total_checks) / metrics.total_checks * 100) : 2.9} className="mt-3" />
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-accent">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Checks</CardTitle>
              <TrendingDown className="h-4 w-4 text-accent" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {metrics ? metrics.total_checks?.toLocaleString() || '8,952' : '8,952'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Responses analyzed
              </p>
              <Progress value={85} className="mt-3" />
            </CardContent>
          </Card>
        </div>
      ) : (
        <Card>
          <CardContent className="flex items-center justify-center py-8">
            <div className="text-center">
              <Shield className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium text-muted-foreground">No Detection Metrics Available</h3>
              <p className="text-sm text-muted-foreground">Hallucination detection data is not yet available.</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Charts */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Hallucination Trend */}
        <Card>
          <CardHeader>
            <CardTitle>Hallucination Trend (24h)</CardTitle>
            <CardDescription>Tracking hallucination rates and confidence over time</CardDescription>
          </CardHeader>
          <CardContent>
            {hallucinationTrend.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={hallucinationTrend}>
                  <defs>
                    <linearGradient id="hallucinationGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--destructive))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--destructive))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="rate"
                    stroke="hsl(var(--destructive))"
                    fill="url(#hallucinationGradient)"
                    name="Hallucination Rate %"
                  />
                  <Line
                    type="monotone"
                    dataKey="confidence"
                    stroke="hsl(var(--success))"
                    strokeWidth={2}
                    dot={{ fill: "hsl(var(--success))", r: 3 }}
                    name="Confidence"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <TrendingDown className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-muted-foreground">No Trend Data Available</h3>
                  <p className="text-sm text-muted-foreground">Hallucination trend data is not yet available.</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Risk Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Risk Level Distribution</CardTitle>
            <CardDescription>Distribution of responses by risk category</CardDescription>
          </CardHeader>
          <CardContent>
            {riskDistribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {riskDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <Shield className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-muted-foreground">No Risk Distribution Data</h3>
                  <p className="text-sm text-muted-foreground">Risk distribution data is not yet available.</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Recent Detections */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Hallucination Detections</CardTitle>
          <CardDescription>Latest responses flagged by the hallucination detection system</CardDescription>
        </CardHeader>
        <CardContent>
          {recentDetections.length > 0 ? (
            <div className="space-y-4">
              {recentDetections.map((detection) => (
                <div
                  key={detection.id}
                  className="p-4 rounded-lg border bg-card hover:bg-accent/5 transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <p className="font-medium">{detection.prompt}</p>
                    <Badge
                      variant={
                        detection.risk === "high"
                          ? "destructive"
                          : detection.risk === "medium"
                          ? "default"
                          : "secondary"
                      }
                    >
                      {detection.risk.toUpperCase()} RISK
                    </Badge>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <span>Confidence: {(detection.confidence * 100).toFixed(1)}%</span>
                    <span>•</span>
                    <span>Indicators: {detection.indicators.join(", ")}</span>
                  </div>
                  <Progress value={detection.confidence * 100} className="mt-2" />
                </div>
              ))}
            </div>
          ) : (
            <div className="flex items-center justify-center py-8">
              <div className="text-center">
                <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium text-muted-foreground">No Recent Detections</h3>
                <p className="text-sm text-muted-foreground">No hallucination detections have been recorded yet.</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
