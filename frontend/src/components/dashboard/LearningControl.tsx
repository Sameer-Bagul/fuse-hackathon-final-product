import { useState, useEffect } from "react";
import { Send, Square, MessageSquare, Loader2, Trash2, AlertCircle, CheckCircle, Clock, Zap } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { apiService, PromptResponse } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";

interface LoopStatus {
  is_running: boolean;
  waiting_for_initial_prompt: boolean;
  initial_prompt_received: boolean;
  current_iteration: number;
  initial_prompt_data?: {
    prompt_text: string;
    user_id: string;
    timestamp: number;
  };
  message: string;
}

export const LearningControl = () => {
  const [userPrompt, setUserPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processingPrompt, setProcessingPrompt] = useState(false);
  const [lastResponse, setLastResponse] = useState<PromptResponse | null>(null);
  const [isLearningActive, setIsLearningActive] = useState(false);
  const [clearingData, setClearingData] = useState(false);
  const [loopStatus, setLoopStatus] = useState<LoopStatus | null>(null);

  // Fetch loop status on mount and periodically
  useEffect(() => {
    fetchLoopStatus();
    const interval = setInterval(fetchLoopStatus, 3000); // Check every 3 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    console.log(`Learning process ${isLearningActive ? 'started' : 'stopped'}`);
  }, [isLearningActive]);

  const fetchLoopStatus = async () => {
    try {
      const response = await fetch('http://localhost:8082/learning/autonomous/status');
      const data = await response.json();
      setLoopStatus(data);
      setIsLearningActive(data.is_running);
    } catch (err) {
      console.error('Failed to fetch loop status:', err);
    }
  };

  const handleProcessPrompt = async () => {
    if (!userPrompt.trim()) {
      toast.error("Please enter a prompt to process");
      return;
    }

    try {
      setProcessingPrompt(true);
      const request = {
        prompt_text: userPrompt
      };

      const response = await apiService.processUserPrompt(request, "user_dashboard");
      setLastResponse(response);

      // The backend will automatically trigger the loop on first prompt
      toast.success("Initial prompt submitted! Autonomous learning will start automatically.");
      
      // Wait a moment and refresh status
      setTimeout(() => {
        fetchLoopStatus();
      }, 1000);

      setUserPrompt(""); // Clear the input after successful processing

    } catch (err) {
      toast.error("Failed to process prompt");
      console.error('Error processing prompt:', err);
    } finally {
      setProcessingPrompt(false);
    }
  };

  const handleStopLearning = async () => {
    try {
      console.log('Attempting to stop autonomous learning loop');
      const result = await apiService.stopAutonomousLearning();
      if (result.success) {
        console.log('Autonomous learning loop stopped successfully');
        setIsLearningActive(false);
        toast.success("Learning loop stopped");
        fetchLoopStatus(); // Refresh status
      } else {
        console.log('Autonomous learning loop stop failed: result not successful');
        toast.warning("Learning loop may not be running or failed to stop");
      }
    } catch (err) {
      console.log('Error stopping learning:', err);
      toast.warning("Learning loop may not be running or failed to stop");
      console.error('Error stopping learning:', err);
    }
  };

  const handleClearAllData = async () => {
    try {
      setClearingData(true);
      console.log('Attempting to clear all learning data');

      // First stop any active learning
      if (isLearningActive) {
        await apiService.stopAutonomousLearning();
        setIsLearningActive(false);
      }

      // Reset all learning data
      const result = await apiService.resetAllLearningData(true);
      if (result.success) {
        console.log('All learning data cleared successfully');
        toast.success("All learning data has been cleared. Start fresh!");
        setLastResponse(null);
        setUserPrompt("");
      } else {
        console.log('Failed to clear learning data');
        toast.error("Failed to clear learning data");
      }
    } catch (err) {
      console.log('Error clearing data:', err);
      toast.error("Failed to clear learning data");
      console.error('Error clearing data:', err);
    } finally {
      setClearingData(false);
    }
  };


  if (error) {
    return (
      <Alert className="border-destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Failed to load learning control: {error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
  <div className="space-y-6">
      {/* Loop Status Banner */}
      {loopStatus && (
        <Alert className={
          loopStatus.waiting_for_initial_prompt 
            ? "border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20" 
            : loopStatus.is_running 
            ? "border-green-500 bg-green-50 dark:bg-green-950/20"
            : "border-gray-300 bg-gray-50 dark:bg-gray-950/20"
        }>
          <div className="flex items-start gap-3">
            {loopStatus.waiting_for_initial_prompt ? (
              <Clock className="h-5 w-5 text-yellow-600 mt-0.5" />
            ) : loopStatus.is_running ? (
              <Zap className="h-5 w-5 text-green-600 mt-0.5 animate-pulse" />
            ) : (
              <AlertCircle className="h-5 w-5 text-gray-600 mt-0.5" />
            )}
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <p className="font-semibold text-sm">
                  {loopStatus.waiting_for_initial_prompt ? "Waiting for Initial Prompt" : 
                   loopStatus.is_running ? "Autonomous Learning Active" : 
                   "Learning Loop Stopped"}
                </p>
                {loopStatus.is_running && (
                  <Badge variant="outline" className="bg-green-100 text-green-700 border-green-300">
                    Iteration {loopStatus.current_iteration}
                  </Badge>
                )}
              </div>
              <p className="text-sm text-muted-foreground">
                {loopStatus.message}
              </p>
              {loopStatus.initial_prompt_data && (
                <div className="mt-2 p-2 bg-white/50 dark:bg-black/20 rounded border">
                  <p className="text-xs font-medium text-muted-foreground mb-1">Initial Prompt:</p>
                  <p className="text-xs italic">"{loopStatus.initial_prompt_data.prompt_text.substring(0, 100)}..."</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    by {loopStatus.initial_prompt_data.user_id} • 
                    {new Date(loopStatus.initial_prompt_data.timestamp * 1000).toLocaleString()}
                  </p>
                </div>
              )}
            </div>
          </div>
        </Alert>
      )}

      {/* Learning Control */}
      <Card className="border-t-4 border-t-primary">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Learning Control
          </CardTitle>
          <CardDescription>
            {loopStatus?.waiting_for_initial_prompt ? (
              <span className="text-yellow-600 dark:text-yellow-400 font-medium">
                Submit your first prompt to begin autonomous learning! The system will then generate all subsequent prompts automatically.
              </span>
            ) : loopStatus?.is_running ? (
              <span className="text-green-600 dark:text-green-400 font-medium">
                LLM is generating prompts autonomously. All iterations are happening in the background.
              </span>
            ) : (
              "Enter an initial prompt to start autonomous learning. The system will learn and improve continuously."
            )}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {loopStatus?.waiting_for_initial_prompt && (
              <div className="p-4 bg-blue-50 dark:bg-blue-950/20 border-2 border-blue-200 dark:border-blue-800 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-1">
                      How User-Driven Learning Works:
                    </p>
                    <ol className="text-sm text-blue-800 dark:text-blue-200 space-y-1 list-decimal list-inside">
                      <li><strong>You</strong> submit the <strong>first prompt</strong> (below)</li>
                      <li><strong>System</strong> processes it and starts autonomous loop</li>
                      <li><strong>LLM</strong> generates <strong>all subsequent prompts</strong> automatically</li>
                      <li><strong>System</strong> learns continuously without further input</li>
                    </ol>
                  </div>
                </div>
              </div>
            )}

            <div className="relative">
              <Textarea
                placeholder={
                  loopStatus?.waiting_for_initial_prompt 
                    ? "Enter your initial prompt here to kickstart autonomous learning... (e.g., 'Teach me about Python basics and ML fundamentals')"
                    : loopStatus?.is_running
                    ? "Autonomous learning is active. LLM is generating prompts automatically..."
                    : "Enter a new prompt to restart autonomous learning..."
                }
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                disabled={processingPrompt || isLearningActive}
                className="min-h-[120px] resize-none pr-20"
              />
              {loopStatus?.waiting_for_initial_prompt && (
                <Badge className="absolute top-3 right-3 bg-yellow-100 text-yellow-800 border-yellow-300">
                  Initial Prompt Required
                </Badge>
              )}
              {loopStatus?.is_running && (
                <Badge className="absolute top-3 right-3 bg-green-100 text-green-800 border-green-300">
                  Loop Running
                </Badge>
              )}
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Button
                  onClick={handleProcessPrompt}
                  disabled={!userPrompt.trim() || processingPrompt || isLearningActive}
                  className="bg-gradient-primary"
                  size="lg"
                >
                  {processingPrompt ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Processing Initial Prompt...
                    </>
                  ) : loopStatus?.waiting_for_initial_prompt ? (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      Submit Initial Prompt & Start Learning
                    </>
                  ) : (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      Restart with New Prompt
                    </>
                  )}
                </Button>
                <Button
                  onClick={handleStopLearning}
                  variant="destructive"
                  size="lg"
                  className="bg-red-600 hover:bg-red-700"
                  disabled={processingPrompt}
                >
                  <Square className="h-4 w-4 mr-2" />
                  Stop Learning
                </Button>
                <Button
                  onClick={handleClearAllData}
                  variant="outline"
                  size="lg"
                  className="border-red-300 text-red-600 hover:bg-red-50 hover:border-red-400"
                  disabled={clearingData || processingPrompt}
                >
                  {clearingData ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Clearing...
                    </>
                  ) : (
                    <>
                      <Trash2 className="h-4 w-4 mr-2" />
                      Clear All Data
                    </>
                  )}
                </Button>
                <span className="text-sm text-muted-foreground">
                  {userPrompt.length} characters
                </span>
              </div>
            </div>
            {isLearningActive && loopStatus && (
              <div className="p-4 bg-green-50 dark:bg-green-950/20 border-2 border-green-200 dark:border-green-800 rounded-lg">
                <div className="flex items-start gap-3">
                  <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm text-green-800 dark:text-green-200 font-semibold mb-2">
                      Autonomous Learning Active - LLM is in Control
                    </p>
                    <div className="grid grid-cols-2 gap-4 text-xs">
                      <div>
                        <p className="text-green-700 dark:text-green-300 font-medium">Current Status:</p>
                        <p className="text-green-600 dark:text-green-400">• Iteration #{loopStatus.current_iteration}</p>
                        <p className="text-green-600 dark:text-green-400">• LLM generating prompts automatically</p>
                        <p className="text-green-600 dark:text-green-400">• Learning from each iteration</p>
                      </div>
                      <div>
                        <p className="text-green-700 dark:text-green-300 font-medium">What's Happening:</p>
                        <p className="text-green-600 dark:text-green-400">• System generates new prompts (1/sec)</p>
                        <p className="text-green-600 dark:text-green-400">• Processes through curriculum</p>
                        <p className="text-green-600 dark:text-green-400">• Continuous improvement via PPO & Q-learning</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
