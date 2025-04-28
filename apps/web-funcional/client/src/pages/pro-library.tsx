import { Sidebar } from "@/components/sidebar";
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Bell, HelpCircle, Search, ChevronRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function ProLibrary() {
  const [searchTerm, setSearchTerm] = useState("");
  
  const { data: proPlayers = [] } = useQuery({
    queryKey: ["/api/pro-players"],
  });

  const filteredPlayers = proPlayers.filter((player: any) => 
    player.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    player.specialties.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Mock video categories
  const categories = [
    "Forehand",
    "Backhand",
    "Serve",
    "Volley",
    "Footwork",
    "All Shots"
  ];

  return (
    <div className="min-h-screen flex flex-col md:flex-row">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="px-6 py-4 flex justify-between items-center">
            <h1 className="text-2xl font-bold text-neutral-800">Pro Players Library</h1>
            <div className="flex items-center">
              <Button variant="ghost" size="icon" className="text-neutral-400 mr-4">
                <Bell className="h-5 w-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-neutral-400">
                <HelpCircle className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </header>

        <div className="p-6">
          <Card className="mb-8">
            <CardContent className="p-6">
              <h2 className="text-xl font-semibold mb-4">Find Professional Technique</h2>
              <div className="relative mb-6">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400 h-4 w-4" />
                <Input 
                  placeholder="Search by player name or shot type..."
                  className="pl-9"
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                />
              </div>
              
              <Tabs defaultValue="All Shots">
                <TabsList className="mb-6">
                  {categories.map(category => (
                    <TabsTrigger key={category} value={category}>
                      {category}
                    </TabsTrigger>
                  ))}
                </TabsList>
                
                {categories.map(category => (
                  <TabsContent key={category} value={category}>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {filteredPlayers.map((player: any) => (
                        <Card key={player.id}>
                          <CardContent className="p-0">
                            <div className="aspect-video bg-neutral-200 flex items-center justify-center text-3xl font-bold text-neutral-400">
                              {player.name.substring(0, 2)}
                            </div>
                            <div className="p-4">
                              <div className="flex justify-between items-center">
                                <div>
                                  <h3 className="font-semibold text-lg">{player.name}</h3>
                                  <p className="text-xs text-neutral-400">{player.specialties}</p>
                                </div>
                                <Button variant="ghost" size="icon" className="text-primary">
                                  <ChevronRight className="h-5 w-5" />
                                </Button>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </TabsContent>
                ))}
              </Tabs>
            </CardContent>
          </Card>
          
          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-4">Featured Technique Videos</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[1, 2, 3].map((i) => (
                <Card key={i} className="overflow-hidden">
                  <div className="aspect-video bg-neutral-200 flex items-center justify-center">
                    <Button 
                      variant="outline" 
                      size="icon" 
                      className="w-12 h-12 rounded-full bg-white/80 hover:bg-white/90"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5 text-primary">
                        <polygon points="5 3 19 12 5 21 5 3" />
                      </svg>
                    </Button>
                  </div>
                  <CardContent className="p-4">
                    {i === 1 && <h3 className="font-semibold">Federer's One-Handed Backhand</h3>}
                    {i === 2 && <h3 className="font-semibold">Nadal's Topspin Forehand</h3>}
                    {i === 3 && <h3 className="font-semibold">Djokovic's Return Technique</h3>}
                    <p className="text-xs text-neutral-400 mt-1">
                      {i === 1 && "Perfect technique breakdown"}
                      {i === 2 && "Heavy topspin generation"}
                      {i === 3 && "Positioning and timing"}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
