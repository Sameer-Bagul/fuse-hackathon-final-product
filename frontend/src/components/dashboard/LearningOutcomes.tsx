import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { ChevronDown, ChevronRight, MessageSquare, Bot, Award, Target, Clock, AlertTriangle, TrendingUp, Activity, Zap, Brain } from "lucide-react";
import { apiService } from "@/lib/api";

interface FeedbackInfluence {
  collaborative_insights?: number;
  user_preferences?: boolean;
  correction_patterns?: boolean;
}

interface LearningOutcome {
  iteration: number;
  timestamp: number;
  prompt: string;
  response: string;
  action: number;
  reward: number;
  target_skill: string;
  hallucination_detected: boolean;
  processing_time: number;
  feedback_influence?: FeedbackInfluence;
}

export const LearningOutcomes = () => {
  const [outcomes, setOutcomes] = useState<LearningOutcome[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedItems, setExpandedItems] = useState<Set<number>>(new Set());

  useEffect(() => {
    const fetchOutcomes = async () => {
      try {
        setLoading(true);
        const progress = await apiService.getLearningProgress();
        if (progress.learning_outcomes) {
          setOutcomes(progress.learning_outcomes);
        }
      } catch (error) {
        console.error('Error fetching learning outcomes:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchOutcomes();

    // Refresh outcomes every 5 seconds if learning is active
    const interval = setInterval(fetchOutcomes, 5000);
    return () => clearInterval(interval);
  }, []);

  const toggleExpanded = (iteration: number) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(iteration)) {
      newExpanded.delete(iteration);
    } else {
      newExpanded.add(iteration);
    }
    setExpandedItems(newExpanded);
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  const getRewardColor = (reward: number) => {
    if (reward >= 0.8) return "text-green-600";
    if (reward >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  // Compute metrics from outcomes
  const computeMetrics = () => {
    if (outcomes.length === 0) return null;

    const totalOutcomes = outcomes.length;
    const avgReward = outcomes.reduce((sum, o) => sum + o.reward, 0) / totalOutcomes;
    const hallucinationRate = (outcomes.filter(o => o.hallucination_detected).length / totalOutcomes) * 100;
    const avgProcessingTime = outcomes.reduce((sum, o) => sum + (o.processing_time || 0), 0) / totalOutcomes;

    return {
      totalOutcomes,
      avgReward,
      hallucinationRate,
      avgProcessingTime
    };
  };

  // Prepare chart data
  const prepareChartData = () => {
    if (outcomes.length === 0) return { rewardData: [], timeData: [] };

    const rewardData = outcomes.slice().reverse().map((outcome, index) => ({
      iteration: outcome.iteration,
      reward: outcome.reward,
      time: formatTime(outcome.timestamp)
    }));

    const timeData = outcomes.slice().reverse().map((outcome, index) => ({
      iteration: outcome.iteration,
      processingTime: outcome.processing_time || 0,
      time: formatTime(outcome.timestamp)
    }));

    return { rewardData, timeData };
  };

  const metrics = computeMetrics();
  const { rewardData, timeData } = prepareChartData();

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Learning Outcomes</CardTitle>
          <CardDescription>Recent learning iterations and their results</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-muted rounded w-1/2"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }


  return (
    <div className="space-y-6">
      {/* Metrics Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="border-l-4 border-l-primary">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Outcomes</CardTitle>
            <Target className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics ? metrics.totalOutcomes : 'No data'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Learning iterations completed
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-success">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Reward</CardTitle>
            <Award className="h-4 w-4 text-success" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics ? metrics.avgReward.toFixed(2) : 'No data'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Performance score
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-warning">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Hallucination Rate</CardTitle>
            <AlertTriangle className="h-4 w-4 text-warning" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics ? `${metrics.hallucinationRate.toFixed(1)}%` : 'No data'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Inaccurate responses
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-accent">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-accent" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics ? `${metrics.avgProcessingTime.toFixed(2)}s` : 'No data'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Response time
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Reward Trends</CardTitle>
            <CardDescription>Reward scores over learning iterations</CardDescription>
          </CardHeader>
          <CardContent>
            {rewardData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={rewardData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="iteration" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="reward"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    dot={{ fill: "hsl(var(--primary))", r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
                <TrendingUp className="h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold text-muted-foreground mb-2">No Reward Data</h3>
                <p className="text-sm text-muted-foreground">Reward trends will appear here once learning outcomes are available.</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Processing Time</CardTitle>
            <CardDescription>Response processing time over iterations</CardDescription>
          </CardHeader>
          <CardContent>
            {timeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={timeData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="iteration" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="processingTime" fill="hsl(var(--accent))" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
                <Activity className="h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold text-muted-foreground mb-2">No Processing Data</h3>
                <p className="text-sm text-muted-foreground">Processing time metrics will appear here once learning outcomes are available.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Learning Outcomes List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Detailed Outcomes
          </CardTitle>
          <CardDescription>
            Recent learning iterations and their results ({outcomes.length} outcomes)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[600px] pr-4">
            <div className="space-y-4">
              {outcomes.length > 0 ? outcomes.slice().reverse().map((outcome) => (
                <Card key={outcome.iteration} className="border-l-4 border-l-primary/50">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">Iteration {outcome.iteration}</Badge>
                        <div className="flex items-center gap-1 text-xs text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          {formatTime(outcome.timestamp)}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={outcome.reward >= 0.6 ? "default" : "secondary"}
                          className={getRewardColor(outcome.reward)}
                        >
                          <Award className="h-3 w-3 mr-1" />
                          {outcome.reward.toFixed(2)}
                        </Badge>
                        {outcome.hallucination_detected && (
                          <Badge variant="destructive" className="text-xs">
                            <AlertTriangle className="h-3 w-3 mr-1" />
                            Hallucination
                          </Badge>
                        )}
                      </div>
                    </div>

                    <div className="space-y-2 mb-3">
                      <div className="flex items-center gap-2 text-sm">
                        <Target className="h-4 w-4 text-primary" />
                        <span className="font-medium">Skill:</span>
                        <Badge variant="outline">{outcome.target_skill || 'General'}</Badge>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <Bot className="h-4 w-4 text-primary" />
                        <span className="font-medium">Action:</span>
                        <Badge variant="outline">{outcome.action}</Badge>
                        {outcome.processing_time && (
                          <span className="text-xs text-muted-foreground">
                            ({outcome.processing_time.toFixed(2)}s)
                          </span>
                        )}
                      </div>
                    </div>

                    <Collapsible
                      open={expandedItems.has(outcome.iteration)}
                      onOpenChange={() => toggleExpanded(outcome.iteration)}
                    >
                      <CollapsibleTrigger asChild>
                        <Button variant="ghost" size="sm" className="w-full justify-between p-2 h-auto">
                          <span className="flex items-center gap-2">
                            <MessageSquare className="h-4 w-4" />
                            View Details
                          </span>
                          {expandedItems.has(outcome.iteration) ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                        </Button>
                      </CollapsibleTrigger>
                      <CollapsibleContent className="space-y-3 mt-3">
                        <div className="space-y-2">
                          <h5 className="text-sm font-medium text-primary">Prompt:</h5>
                          <div className="p-3 bg-muted rounded-md">
                            <p className="text-sm">{outcome.prompt}</p>
                          </div>
                        </div>

                        <div className="space-y-2">
                          <h5 className="text-sm font-medium text-primary">Response:</h5>
                          <div className="p-3 bg-muted rounded-md">
                            <p className="text-sm whitespace-pre-wrap">{outcome.response}</p>
                          </div>
                        </div>

                        {outcome.feedback_influence && (
                          <div className="space-y-2">
                            <h5 className="text-sm font-medium text-primary">Feedback Influence:</h5>
                            <div className="flex flex-wrap gap-1">
                              {outcome.feedback_influence.collaborative_insights > 0 && (
                                <Badge variant="outline" className="text-xs">
                                  Collaborative Insights ({outcome.feedback_influence.collaborative_insights})
                                </Badge>
                              )}
                              {outcome.feedback_influence.user_preferences && (
                                <Badge variant="outline" className="text-xs">
                                  User Preferences
                                </Badge>
                              )}
                              {outcome.feedback_influence.correction_patterns && (
                                <Badge variant="outline" className="text-xs">
                                  Correction Patterns
                                </Badge>
                              )}
                            </div>
                          </div>
                        )}
                      </CollapsibleContent>
                    </Collapsible>
                  </CardContent>
                </Card>
              )) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No learning outcomes available yet.</p>
                  <p className="text-sm">Start the learning loop to see outcomes here.</p>
                </div>
              )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  </div>
);
};