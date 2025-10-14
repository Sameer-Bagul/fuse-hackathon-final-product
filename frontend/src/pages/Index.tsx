import { useState, useEffect } from "react";
import { Brain, Activity, Target, AlertTriangle, Award, Database, Zap, History, Square, Play } from "lucide-react";
import { MetricsOverview } from "@/components/dashboard/MetricsOverview";
import { LearningControl } from "@/components/dashboard/LearningControl";
import { HallucinationMonitor } from "@/components/dashboard/HallucinationMonitor";
import { RewardSystem } from "@/components/dashboard/RewardSystem";
import { CurriculumProgress } from "@/components/dashboard/CurriculumProgress";
import { AnalyticsDashboard } from "@/components/dashboard/AnalyticsDashboard";
import { LearningHistory } from "@/components/dashboard/LearningHistory";
import { ConnectionIndicator } from "@/components/ui/ConnectionIndicator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { apiService } from "@/lib/api";
import { toast } from "sonner";

const Index = () => {
  const [activeTab, setActiveTab] = useState("overview");
  const [learningLoopRunning, setLearningLoopRunning] = useState(false);
  const [stoppingLoop, setStoppingLoop] = useState(false);

  // Check learning loop status periodically
  useEffect(() => {
    const checkLearningLoopStatus = async () => {
      try {
        const progress = await apiService.getLearningLoopProgress();
        // Assuming the progress response has an 'is_running' field or similar
        // For now, we'll check if iterations > 0 as a proxy
        setLearningLoopRunning(progress.iteration_count > 0);
      } catch (err) {
        console.error('Error checking learning loop status:', err);
        setLearningLoopRunning(false);
      }
    };

    // Check immediately and then every 5 seconds
    checkLearningLoopStatus();
    const interval = setInterval(checkLearningLoopStatus, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleStopLearningLoop = async () => {
    try {
      setStoppingLoop(true);
      await apiService.stopLearning();
      setLearningLoopRunning(false);
      toast.success("Autonomous learning loop stopped successfully!");
    } catch (err) {
      toast.error("Failed to stop learning loop");
      console.error('Error stopping learning loop:', err);
    } finally {
      setStoppingLoop(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-primary">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                  ML Learning System
                </h1>
                <p className="text-sm text-muted-foreground">Advanced LLM Monitoring & Control</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {learningLoopRunning && (
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-sm font-medium">Learning Active</span>
                  </div>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={handleStopLearningLoop}
                    disabled={stoppingLoop}
                    className="gap-2"
                  >
                    {stoppingLoop ? (
                      <>
                        <Square className="h-4 w-4 animate-spin" />
                        Stopping...
                      </>
                    ) : (
                      <>
                        <Square className="h-4 w-4" />
                        Stop Learning
                      </>
                    )}
                  </Button>
                </div>
              )}
              <ConnectionIndicator />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-8 lg:w-auto lg:inline-grid">
            <TabsTrigger value="overview" className="gap-2">
              <Activity className="h-4 w-4" />
              <span className="hidden sm:inline">Overview</span>
            </TabsTrigger>
            <TabsTrigger value="control" className="gap-2">
              <Zap className="h-4 w-4" />
              <span className="hidden sm:inline">Control</span>
            </TabsTrigger>
            <TabsTrigger value="hallucination" className="gap-2">
              <AlertTriangle className="h-4 w-4" />
              <span className="hidden sm:inline">Hallucination</span>
            </TabsTrigger>
            <TabsTrigger value="rewards" className="gap-2">
              <Award className="h-4 w-4" />
              <span className="hidden sm:inline">Rewards</span>
            </TabsTrigger>
            <TabsTrigger value="curriculum" className="gap-2">
              <Target className="h-4 w-4" />
              <span className="hidden sm:inline">Curriculum</span>
            </TabsTrigger>
            <TabsTrigger value="history" className="gap-2">
              <History className="h-4 w-4" />
              <span className="hidden sm:inline">History</span>
            </TabsTrigger>
            <TabsTrigger value="analytics" className="gap-2">
              <Database className="h-4 w-4" />
              <span className="hidden sm:inline">Analytics</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <MetricsOverview />
          </TabsContent>

          <TabsContent value="control" className="space-y-6">
            <LearningControl />
          </TabsContent>

          <TabsContent value="hallucination" className="space-y-6">
            <HallucinationMonitor />
          </TabsContent>

          <TabsContent value="rewards" className="space-y-6">
            <RewardSystem />
          </TabsContent>

          <TabsContent value="curriculum" className="space-y-6">
            <CurriculumProgress />
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <LearningHistory />
          </TabsContent>

          <TabsContent value="analytics" className="space-y-6">
            <AnalyticsDashboard />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Index;
