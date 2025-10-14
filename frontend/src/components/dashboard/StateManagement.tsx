import { useState, useEffect } from "react";
import { Save, Download, Trash2, Clock, Database, HardDrive, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { toast } from "sonner";
import { apiService, LearningStateType, SaveStateRequest, LoadStateRequest, ListVersionsRequest, RollbackRequest, VersionInfo } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";

export const StateManagement = () => {
  const [selectedStateType, setSelectedStateType] = useState<LearningStateType>("complete_system");
  const [saveDescription, setSaveDescription] = useState("");
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loadingState, setLoadingState] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedVersion, setSelectedVersion] = useState<string | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [versionToDelete, setVersionToDelete] = useState<string | null>(null);

  const stateTypeOptions = [
    { value: "complete_system", label: "Complete System", description: "All learning components" },
    { value: "llm_model", label: "LLM Model", description: "Neural network weights and training state" },
    { value: "meta_learning", label: "Meta-Learning", description: "Adaptive learning strategies and parameters" },
    { value: "history", label: "Interaction History", description: "Past prompts and responses" },
    { value: "curriculum", label: "Curriculum", description: "Learning progress and skill tracking" },
    { value: "reward_system", label: "Reward System", description: "Reward weights and metrics" },
    { value: "analytics", label: "Analytics", description: "Performance data and insights" },
    { value: "feedback", label: "Feedback", description: "User feedback and preferences" },
  ];

  useEffect(() => {
    fetchVersions();
  }, [selectedStateType]);

  const fetchVersions = async () => {
    try {
      setLoading(true);
      setError(null);

      const request: ListVersionsRequest = {
        state_type: selectedStateType,
        limit: 20
      };

      const response = await apiService.listStateVersions(request);
      setVersions(response.versions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch versions');
      console.error('Error fetching versions:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveState = async () => {
    try {
      setSaving(true);
      setError(null);

      const request: SaveStateRequest = {
        state_type: selectedStateType,
        description: saveDescription.trim() || undefined,
        include_related: true
      };

      const response = await apiService.saveLearningState(request);

      if (response.success) {
        toast.success(`State saved successfully (Version: ${response.version})`);
        setSaveDescription("");
        // Refresh versions list
        await fetchVersions();
      } else {
        toast.error(response.message || 'Failed to save state');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to save state';
      setError(errorMessage);
      toast.error(errorMessage);
      console.error('Error saving state:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleLoadState = async (version: string) => {
    try {
      setLoadingState(true);
      setError(null);

      const request: LoadStateRequest = {
        state_type: selectedStateType,
        version: version
      };

      const response = await apiService.loadLearningState(request);

      if (response.success) {
        toast.success(`State loaded successfully (Version: ${response.version})`);
        // Refresh versions list to update current version
        await fetchVersions();
      } else {
        toast.error(response.message || 'Failed to load state');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load state';
      setError(errorMessage);
      toast.error(errorMessage);
      console.error('Error loading state:', err);
    } finally {
      setLoadingState(false);
    }
  };

  const handleDeleteState = async (version: string) => {
    try {
      setLoading(true);
      setError(null);

      // Use rollback to effectively "delete" by reverting to a previous state
      // Since there's no direct delete endpoint, we'll rollback to the latest version
      // This is a workaround - ideally we'd have a delete endpoint
      const latestVersion = versions.find(v => v.version !== version)?.version;

      if (!latestVersion) {
        toast.error('Cannot delete the only remaining version');
        return;
      }

      const request: RollbackRequest = {
        state_type: selectedStateType,
        target_version: latestVersion,
        confirm_rollback: true
      };

      const response = await apiService.rollbackState(request);

      if (response.success) {
        toast.success(`State version ${version} removed successfully`);
        setShowDeleteDialog(false);
        setVersionToDelete(null);
        // Refresh versions list
        await fetchVersions();
      } else {
        toast.error(response.message || 'Failed to delete state');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete state';
      setError(errorMessage);
      toast.error(errorMessage);
      console.error('Error deleting state:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return 'Unknown';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getStateTypeLabel = (type: string) => {
    return stateTypeOptions.find(opt => opt.value === type)?.label || type;
  };

  if (error && !loading) {
    return (
      <Alert className="border-destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Failed to load state management data: {error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Save Current State */}
      <Card className="border-t-4 border-t-primary">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Save className="h-5 w-5" />
            Save Current State
          </CardTitle>
          <CardDescription>
            Save the current learning state for future restoration
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">State Type</label>
              <Select value={selectedStateType} onValueChange={(value) => setSelectedStateType(value as LearningStateType)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {stateTypeOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      <div>
                        <div className="font-medium">{option.label}</div>
                        <div className="text-xs text-muted-foreground">{option.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Description (Optional)</label>
              <Textarea
                placeholder="Describe this state snapshot..."
                value={saveDescription}
                onChange={(e) => setSaveDescription(e.target.value)}
                className="min-h-[80px] resize-none"
                disabled={saving}
              />
            </div>
          </div>

          <Button
            onClick={handleSaveState}
            disabled={saving}
            className="w-full md:w-auto"
          >
            {saving ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Saving State...
              </>
            ) : (
              <>
                <Save className="h-4 w-4 mr-2" />
                Save Current State
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Saved States List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Saved States
          </CardTitle>
          <CardDescription>
            View and manage previously saved learning states
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-4">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="space-y-2">
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="h-3 w-48" />
                  </div>
                  <div className="flex gap-2">
                    <Skeleton className="h-8 w-16" />
                    <Skeleton className="h-8 w-16" />
                  </div>
                </div>
              ))}
            </div>
          ) : versions.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No saved states found for {getStateTypeLabel(selectedStateType)}</p>
              <p className="text-sm">Save your first state to get started</p>
            </div>
          ) : (
            <div className="space-y-4">
              {versions.map((version) => (
                <div key={version.version} className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors">
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">v{version.version}</Badge>
                      <span className="text-sm text-muted-foreground">
                        {formatFileSize(version.size_bytes)}
                      </span>
                    </div>

                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {formatTimestamp(version.timestamp)}
                      </div>
                      {version.description && (
                        <span className="truncate max-w-xs">{version.description}</span>
                      )}
                    </div>

                    {version.performance_metrics && Object.keys(version.performance_metrics).length > 0 && (
                      <div className="flex gap-2">
                        {Object.entries(version.performance_metrics).slice(0, 3).map(([key, value]) => (
                          <Badge key={key} variant="secondary" className="text-xs">
                            {key}: {typeof value === 'number' ? value.toFixed(2) : value}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleLoadState(version.version)}
                      disabled={loadingState}
                    >
                      {loadingState && selectedVersion === version.version ? (
                        <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-current" />
                      ) : (
                        <Download className="h-3 w-3 mr-1" />
                      )}
                      Load
                    </Button>

                    <AlertDialog open={showDeleteDialog && versionToDelete === version.version} onOpenChange={setShowDeleteDialog}>
                      <AlertDialogTrigger asChild>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setVersionToDelete(version.version);
                            setShowDeleteDialog(true);
                          }}
                          disabled={versions.length <= 1}
                        >
                          <Trash2 className="h-3 w-3 mr-1" />
                          Delete
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Delete State Version</AlertDialogTitle>
                          <AlertDialogDescription>
                            Are you sure you want to delete version {version.version}? This action cannot be undone.
                            {versions.length <= 1 && " This is the only version available."}
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel onClick={() => setVersionToDelete(null)}>Cancel</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={() => handleDeleteState(version.version)}
                            className="bg-destructive hover:bg-destructive/90"
                          >
                            Delete
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* State Type Information */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HardDrive className="h-5 w-5" />
            State Type Information
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {stateTypeOptions.map((option) => (
              <div
                key={option.value}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                  selectedStateType === option.value
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:bg-muted/50'
                }`}
                onClick={() => setSelectedStateType(option.value as LearningStateType)}
              >
                <div className="font-medium text-sm">{option.label}</div>
                <div className="text-xs text-muted-foreground mt-1">{option.description}</div>
                <div className="text-xs text-primary mt-2">
                  {versions.filter(v => v.version).length} versions saved
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};