import React, { useState, useEffect } from 'react';
import { apiService } from '../../lib/api';

interface ConnectionIndicatorProps {
  className?: string;
}

export const ConnectionIndicator: React.FC<ConnectionIndicatorProps> = ({ className = '' }) => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isChecking, setIsChecking] = useState<boolean>(false);

  const checkConnection = async () => {
    setIsChecking(true);
    try {
      const healthResponse = await apiService.checkHealth();
      setIsConnected(healthResponse.status === 'healthy');
    } catch (error) {
      console.warn('Connection check failed:', error);
      setIsConnected(false);
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    // Initial check
    checkConnection();

    // Set up polling every 30 seconds
    const interval = setInterval(checkConnection, 30000);

    // Cleanup interval on unmount
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <div
        className={`w-3 h-3 rounded-full ${
          isConnected ? 'bg-green-500' : 'bg-red-500'
        } ${isChecking ? 'animate-pulse' : ''}`}
        title={isConnected ? 'Server connected' : 'Server disconnected'}
      />
      <span className="text-sm text-gray-600">
        {isConnected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  );
};