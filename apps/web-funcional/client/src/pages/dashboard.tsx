import { Sidebar } from "@/components/sidebar";
import { UploadSection } from "@/components/dashboard/upload-section";
import { AnalysisSection } from "@/components/dashboard/analysis-section";
import { ProgressSection } from "@/components/dashboard/progress-section";
import { Bell, HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Dashboard() {
  return (
    <div className="min-h-screen flex flex-col md:flex-row">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="px-6 py-4 flex justify-between items-center">
            <h1 className="text-2xl font-bold text-neutral-800">Dashboard</h1>
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
          <UploadSection />
          <AnalysisSection />
          <ProgressSection />
        </div>
      </main>
    </div>
  );
}
