import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle, History } from "lucide-react";
import { apiService, LearningHistoryResponse, LearningHistoryItem } from "@/lib/api";

export const LearningHistory = () => {
  const [historyData, setHistoryData] = useState<LearningHistoryItem[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        setError(null);

        const response: LearningHistoryResponse = await apiService.getLearningHistory(50, 0);
        setHistoryData(response.interactions || []);
        setTotalCount(response.total_count);

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch learning history');
        console.error('Error fetching learning history:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const formatReward = (reward: number | null) => {
    if (reward === null) return 'N/A';
    return reward.toFixed(3);
  };

  const getRewardBadgeVariant = (reward: number | null) => {
    if (reward === null) return 'secondary';
    if (reward > 0.7) return 'default';
    if (reward > 0.4) return 'secondary';
    return 'destructive';
  };

  const getSourceBadgeVariant = (source: string) => {
    return source === 'user' ? 'default' : 'secondary';
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Learning History
          </CardTitle>
          <CardDescription>Recent LLM prompts, actions, and learning outcomes</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex space-x-4">
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-4 flex-1" />
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-4 w-16" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Learning History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert className="border-destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Failed to load learning history: {error}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="h-5 w-5" />
          Learning History
        </CardTitle>
        <CardDescription>
          Recent LLM prompts, actions, and learning outcomes ({totalCount} total interactions)
        </CardDescription>
      </CardHeader>
      <CardContent>
        {historyData.length === 0 ? (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>No learning history data available yet</AlertDescription>
          </Alert>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Timestamp</TableHead>
                <TableHead>Prompt</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>Reward</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Response</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {historyData.map((item, index) => (
                <TableRow key={index}>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatTimestamp(item.timestamp)}
                  </TableCell>
                  <TableCell className="max-w-xs">
                    <div className="truncate" title={item.prompt_text || 'N/A'}>
                      {item.prompt_text ? (item.prompt_text.length > 50 ? `${item.prompt_text.substring(0, 50)}...` : item.prompt_text) : 'N/A'}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">
                      {item.action !== null ? item.action : 'N/A'}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={getRewardBadgeVariant(item.reward)}>
                      {formatReward(item.reward)}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={getSourceBadgeVariant(item.source)}>
                      {item.source}
                    </Badge>
                  </TableCell>
                  <TableCell className="max-w-xs">
                    <div className="truncate" title={item.response || 'N/A'}>
                      {item.response ? (item.response.length > 50 ? `${item.response.substring(0, 50)}...` : item.response) : 'N/A'}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
};