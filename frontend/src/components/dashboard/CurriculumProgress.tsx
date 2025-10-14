import { Target, BookOpen, Award, TrendingUp, CheckCircle2, Circle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from "recharts";
import { useEffect, useState } from "react";
import { apiService, CurriculumProgressResponse, CurriculumSkillsResponse, CurriculumGapsResponse, CurriculumRecommendationsResponse, Recommendation, CurriculumStatusResponse, CurriculumTaskResponse } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface TaskInfo {
  task_description?: string;
  skill_name?: string;
  difficulty?: string;
  estimated_time?: number;
  learning_objectives?: string[];
  expected_complexity?: number;
}

interface SkillData {
  name: string;
  level: string;
  completed: boolean;
}

interface SkillCategory {
  name: string;
  progress: number;
  skills: SkillData[];
}

interface SkillGap {
  skill: string;
  gap: number;
  priority: string;
}

interface PerformanceRadar {
  category: string;
  value: number;
  fullMark: number;
}

export const CurriculumProgress = () => {
  const [skillCategories, setSkillCategories] = useState<SkillCategory[]>([]);
  const [skillGapData, setSkillGapData] = useState<SkillGap[]>([]);
  const [performanceRadar, setPerformanceRadar] = useState<PerformanceRadar[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [curriculumStatus, setCurriculumStatus] = useState<CurriculumStatusResponse | null>(null);
  const [currentTask, setCurrentTask] = useState<CurriculumTaskResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Use a default learner ID for now
        const learnerId = "default_learner";

        // Fetch curriculum progress
        const progressResponse: CurriculumProgressResponse = await apiService.getCurriculumProgress(learnerId);
        const skillsResponse: CurriculumSkillsResponse = await apiService.getCurriculumSkills();
        const gapsResponse: CurriculumGapsResponse = await apiService.getCurriculumGaps(learnerId);
        const recommendationsResponse: CurriculumRecommendationsResponse = await apiService.getCurriculumRecommendations(learnerId);

        // Process skill categories
        const categories: SkillCategory[] = skillsResponse.skills ? Object.keys(skillsResponse.skills).map(categoryName => {
          const categorySkills = skillsResponse.skills[categoryName];
          const progress = progressResponse.progress?.[categoryName] || 0;
          return {
            name: categoryName,
            progress: progress,
            skills: categorySkills.map((skill) => ({
              name: skill.name,
              level: skill.level,
              completed: skill.completed
            }))
          };
        }) : [
          {
            name: "Language Understanding",
            progress: 85,
            skills: [
              { name: "Sentiment Analysis", level: "Advanced", completed: true },
              { name: "Entity Recognition", level: "Advanced", completed: true },
              { name: "Context Comprehension", level: "Intermediate", completed: false },
            ],
          },
          {
            name: "Reasoning & Logic",
            progress: 72,
            skills: [
              { name: "Deductive Reasoning", level: "Advanced", completed: true },
              { name: "Causal Inference", level: "Intermediate", completed: true },
              { name: "Probabilistic Reasoning", level: "Beginner", completed: false },
            ],
          },
          {
            name: "Creative Generation",
            progress: 68,
            skills: [
              { name: "Story Generation", level: "Intermediate", completed: true },
              { name: "Code Generation", level: "Advanced", completed: true },
              { name: "Creative Problem Solving", level: "Beginner", completed: false },
            ],
          },
          {
            name: "Factual Knowledge",
            progress: 91,
            skills: [
              { name: "Historical Facts", level: "Advanced", completed: true },
              { name: "Scientific Knowledge", level: "Advanced", completed: true },
              { name: "Current Events", level: "Intermediate", completed: true },
            ],
          },
        ];

        setSkillCategories(categories);

        // Process skill gaps
        const gaps: SkillGap[] = gapsResponse.gaps?.map((gap) => ({
          skill: gap.skill_name,
          gap: gap.gap_percentage,
          priority: gap.priority
        })) || [
          { skill: "Probabilistic Reasoning", gap: 45, priority: "High" },
          { skill: "Context Comprehension", gap: 32, priority: "Medium" },
          { skill: "Creative Problem Solving", gap: 28, priority: "Medium" },
          { skill: "Temporal Reasoning", gap: 25, priority: "Low" },
        ];

        setSkillGapData(gaps);

        // Process performance radar
        const radar: PerformanceRadar[] = categories.map(cat => ({
          category: cat.name.split(' ')[0], // Take first word
          value: cat.progress,
          fullMark: 100
        }));

        setPerformanceRadar(radar);

        // Process recommendations
        setRecommendations(recommendationsResponse.recommendations || []);

        // Fetch curriculum learning status
        try {
          const statusResponse: CurriculumStatusResponse = await apiService.getCurriculumStatus();
          setCurriculumStatus(statusResponse);

          // If curriculum is active, get current task
          if (statusResponse.active && statusResponse.next_task) {
            const taskResponse: CurriculumTaskResponse = await apiService.getNextCurriculumTask();
            if (taskResponse.task_info && Object.keys(taskResponse.task_info).length > 0) {
              setCurrentTask(taskResponse);
            }
          }
        } catch (curriculumError) {
          console.warn('Could not fetch curriculum status:', curriculumError);
          setCurriculumStatus(null);
          setCurrentTask(null);
        }

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch curriculum data');
        console.error('Error fetching curriculum data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const overallProgress = skillCategories.length > 0
    ? skillCategories.reduce((sum, cat) => sum + cat.progress, 0) / skillCategories.length
    : 0;

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-4">
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
          Failed to load curriculum data: {error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card className="border-l-4 border-l-primary">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Overall Progress</CardTitle>
            <Target className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              {overallProgress.toFixed(1)}%
            </div>
            <Progress value={overallProgress} className="mt-3" />
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-success">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Skills Mastered</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-success" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-success">
              {skillCategories.reduce((sum, cat) => sum + cat.skills.filter(s => s.completed).length, 0)}/
              {skillCategories.reduce((sum, cat) => sum + cat.skills.length, 0)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {Math.round((skillCategories.reduce((sum, cat) => sum + cat.skills.filter(s => s.completed).length, 0) /
                Math.max(1, skillCategories.reduce((sum, cat) => sum + cat.skills.length, 0))) * 100)}% completion rate
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-accent">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">In Progress</CardTitle>
            <Circle className="h-4 w-4 text-accent" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-accent">
              {skillCategories.reduce((sum, cat) => sum + cat.skills.filter(s => !s.completed).length, 0)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">Currently learning</p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-warning">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Improvement</CardTitle>
            <TrendingUp className="h-4 w-4 text-warning" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-warning">+8.3%</div>
            <p className="text-xs text-muted-foreground mt-1">Per week growth</p>
          </CardContent>
        </Card>
      </div>

      {/* Active Curriculum Learning */}
      {curriculumStatus?.active && (
        <Card className="border-l-4 border-l-purple-500">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-purple-500" />
              Active Curriculum Learning
            </CardTitle>
            <CardDescription>Current curriculum session in progress</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Curriculum Progress */}
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-2">
                  <p className="text-sm font-medium">Tasks Completed</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {curriculumStatus.current_task || 0} / {curriculumStatus.total_tasks || 0}
                  </p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium">Completion</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {curriculumStatus.completion_percentage?.toFixed(1) || 0}%
                  </p>
                  <Progress value={curriculumStatus.completion_percentage || 0} className="mt-2" />
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium">Skills Covered</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {curriculumStatus.skills_covered?.length || 0}
                  </p>
                </div>
              </div>

              {/* Current Task */}
              {currentTask && currentTask.task_info && Object.keys(currentTask.task_info).length > 0 && (
                <div className="p-4 rounded-lg bg-purple-50 dark:bg-purple-950/20 border border-purple-200 dark:border-purple-800">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-purple-900 dark:text-purple-100">
                      Current Task ({currentTask.task_sequence_position} of {currentTask.total_tasks})
                    </h4>
                    <Badge variant="outline" className="bg-purple-100 text-purple-800">
                      {(currentTask.task_info as TaskInfo).difficulty || "Medium"}
                    </Badge>
                  </div>
                  <p className="text-purple-800 dark:text-purple-200 mb-3">
                    {(currentTask.task_info as TaskInfo).task_description || "Complete this learning task"}
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-purple-600 dark:text-purple-400">
                    <div>Skill: {(currentTask.task_info as TaskInfo).skill_name || "Unknown"}</div>
                    <div>Time: {(currentTask.task_info as TaskInfo).estimated_time || 0} min</div>
                    <div>Objectives: {((currentTask.task_info as TaskInfo).learning_objectives || []).length}</div>
                    <div>Complexity: {(currentTask.task_info as TaskInfo).expected_complexity?.toFixed(2) || "N/A"}</div>
                  </div>
                </div>
              )}

              {/* Skills Covered */}
              {curriculumStatus.skills_covered && curriculumStatus.skills_covered.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium">Skills Practiced</p>
                  <div className="flex flex-wrap gap-2">
                    {curriculumStatus.skills_covered.map((skill, index) => (
                      <Badge key={index} variant="secondary" className="bg-purple-100 text-purple-800">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Charts */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Performance Radar */}
        <Card>
          <CardHeader>
            <CardTitle>Skill Category Performance</CardTitle>
            <CardDescription>Performance across major skill categories</CardDescription>
          </CardHeader>
          <CardContent>
            {performanceRadar.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={performanceRadar}>
                  <PolarGrid stroke="hsl(var(--border))" />
                  <PolarAngleAxis dataKey="category" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
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
              <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No performance data available</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Skill Gaps */}
        <Card>
          <CardHeader>
            <CardTitle>Priority Skill Gaps</CardTitle>
            <CardDescription>Skills requiring focused attention</CardDescription>
          </CardHeader>
          <CardContent>
            {skillGapData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={skillGapData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis type="number" domain={[0, 50]} />
                  <YAxis dataKey="skill" type="category" width={150} />
                  <Tooltip />
                  <Bar dataKey="gap" fill="hsl(var(--warning))" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No skill gap data available</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Detailed Skill Categories */}
      <div className="grid gap-6 md:grid-cols-2">
        {skillCategories.length > 0 ? (
          skillCategories.map((category) => (
            <Card key={category.name}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <BookOpen className="h-5 w-5 text-primary" />
                    <CardTitle className="text-lg">{category.name}</CardTitle>
                  </div>
                  <Badge variant="secondary">{category.progress}%</Badge>
                </div>
                <Progress value={category.progress} className="mt-2" />
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {category.skills.map((skill) => (
                    <div
                      key={skill.name}
                      className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        {skill.completed ? (
                          <CheckCircle2 className="h-5 w-5 text-success" />
                        ) : (
                          <Circle className="h-5 w-5 text-muted-foreground" />
                        )}
                        <div>
                          <p className="font-medium">{skill.name}</p>
                          <p className="text-xs text-muted-foreground">{skill.level}</p>
                        </div>
                      </div>
                      {!skill.completed && (
                        <Button size="sm" variant="outline">
                          Continue
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))
        ) : (
          <Card className="md:col-span-2">
            <CardContent className="flex items-center justify-center py-12">
              <div className="text-center text-muted-foreground">
                <BookOpen className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-semibold mb-2">No Curriculum Data</h3>
                <p>Your curriculum tree will appear here once data is available.</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Award className="h-5 w-5" />
            Personalized Recommendations
          </CardTitle>
          <CardDescription>AI-generated learning path suggestions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recommendations.length > 0 ? recommendations.map((rec, index) => (
              <div key={index} className={`p-4 rounded-lg border ${index === 0 ? 'bg-gradient-to-r from-primary/5 to-accent/5' : ''}`}>
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded bg-primary/10">
                    {index === 0 ? <Target className="h-4 w-4 text-primary" /> : <TrendingUp className="h-4 w-4 text-accent" />}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold mb-1">{rec.description || rec.title}</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      {rec.details || rec.explanation}
                    </p>
                    <Button size="sm" variant={index === 0 ? "default" : "outline"}>
                      {rec.action || (index === 0 ? "Start Learning Path" : "Continue")}
                    </Button>
                  </div>
                </div>
              </div>
            )) : (
              <div className="text-center py-8 text-muted-foreground">
                <Award className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-semibold mb-2">No Recommendations Yet</h3>
                <p>Personalized learning recommendations will be generated as you progress.</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
