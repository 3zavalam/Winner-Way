import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";

export function ProgressSection() {
  const [timeRange, setTimeRange] = useState("Last 30 days");
  
  const { data: proPlayers = [] } = useQuery({
    queryKey: ["/api/pro-players"],
  });

  // Mock progress data (in a real app, would come from the backend)
  const progressData = {
    videosAnalyzed: 12,
    weeksTraining: 4,
    techniqueImproved: 28,
    issuesFixed: 7,
    shotImprovement: {
      forehand: 78,
      backhand: 45,
      serve: 62,
      volley: 36,
      footwork: 54
    }
  };

  return (
    <section className="mb-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardContent className="p-6">
          <h2 className="text-xl font-semibold mb-6">Your Progress</h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-neutral-100 rounded-lg p-4 text-center">
              <span className="text-2xl font-bold text-primary block mb-1">{progressData.videosAnalyzed}</span>
              <span className="text-xs text-neutral-400">Videos Analyzed</span>
            </div>
            <div className="bg-neutral-100 rounded-lg p-4 text-center">
              <span className="text-2xl font-bold text-primary block mb-1">{progressData.weeksTraining}</span>
              <span className="text-xs text-neutral-400">Weeks Training</span>
            </div>
            <div className="bg-neutral-100 rounded-lg p-4 text-center">
              <span className="text-2xl font-bold text-primary block mb-1">{progressData.techniqueImproved}%</span>
              <span className="text-xs text-neutral-400">Technique Improved</span>
            </div>
            <div className="bg-neutral-100 rounded-lg p-4 text-center">
              <span className="text-2xl font-bold text-primary block mb-1">{progressData.issuesFixed}</span>
              <span className="text-xs text-neutral-400">Issues Fixed</span>
            </div>
          </div>
          
          <div>
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-semibold">Shot Improvement</h3>
              <select 
                className="bg-neutral-100 border-0 rounded-lg py-1 px-3 text-sm"
                value={timeRange}
                onChange={e => setTimeRange(e.target.value)}
              >
                <option>Last 30 days</option>
                <option>Last 90 days</option>
                <option>All time</option>
              </select>
            </div>
            
            <div className="h-64 w-full bg-neutral-100 rounded-lg flex items-end justify-between px-4 pt-4 pb-0">
              {Object.entries(progressData.shotImprovement).map(([shot, value], index) => (
                <div key={shot} className="relative w-1/5 group">
                  <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-neutral-500 text-white text-xs py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                    {value}%
                  </div>
                  <div className="relative w-full">
                    <div className="absolute bottom-0 left-0 right-0 bg-primary bg-opacity-20 rounded-t-md" style={{ height: "100%" }}></div>
                    <div className="absolute bottom-0 left-0 right-0 bg-primary rounded-t-md" style={{ height: `${value}%` }}></div>
                  </div>
                  <div className="text-center mt-2 text-xs capitalize">{shot}</div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold">Pro Players Library</h2>
            <Link href="/pro-library">
              <a className="text-primary text-sm hover:underline">View All</a>
            </Link>
          </div>
          
          <div className="space-y-4">
            {proPlayers.map((player: any) => (
              <div key={player.id} className="flex items-center p-3 rounded-lg hover:bg-neutral-100 transition-colors duration-200 cursor-pointer">
                <div className="w-16 h-16 bg-neutral-200 rounded-lg mr-4 flex items-center justify-center text-neutral-400">
                  {player.name.substring(0, 2)}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold">{player.name}</h3>
                  <p className="text-xs text-neutral-400">{player.specialties}</p>
                </div>
                <Button variant="ghost" size="icon" className="text-primary">
                  <ChevronRight className="h-5 w-5" />
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </section>
  );
}
