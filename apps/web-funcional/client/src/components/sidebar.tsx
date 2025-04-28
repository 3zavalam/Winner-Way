import { useState } from "react";
import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Menu, Home, Video, LineChart, Users, Settings } from "lucide-react";

interface SidebarProps {
  className?: string;
}

export function Sidebar({ className }: SidebarProps) {
  const [location] = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const navItems = [
    { icon: Home, label: "Dashboard", href: "/" },
    { icon: Video, label: "My Videos", href: "/my-videos" },
    { icon: LineChart, label: "My Progress", href: "/progress" },
    { icon: Users, label: "Pro Library", href: "/pro-library" },
    { icon: Settings, label: "Settings", href: "/settings" },
  ];

  return (
    <aside className={cn("bg-secondary text-white md:w-64 w-full md:h-screen", className)}>
      <div className="p-4 border-b border-blue-800 flex justify-between items-center">
        <div className="flex items-center">
          <span className="text-2xl font-bold">TennisAI</span>
        </div>
        <Button 
          variant="ghost" 
          size="icon" 
          className="md:hidden text-white" 
          onClick={toggleMobileMenu}
        >
          <Menu className="h-6 w-6" />
        </Button>
      </div>
      
      <div className={cn("p-4", isMobileMenuOpen ? "block" : "hidden md:block")}>
        <div className="mb-8">
          <div className="flex items-center mb-4">
            <div className="w-10 h-10 rounded-full bg-blue-700 mr-3 flex items-center justify-center text-white">
              MC
            </div>
            <div>
              <h3 className="font-semibold">Michael Chen</h3>
              <p className="text-xs opacity-70">Premium Member</p>
            </div>
          </div>
        </div>
        
        <nav>
          <ul className="space-y-1">
            {navItems.map((item) => {
              const isActive = location === item.href;
              return (
                <li key={item.href}>
                  <Link href={item.href}>
                    <div className={cn(
                      "flex items-center p-3 rounded-lg transition-colors duration-200 cursor-pointer",
                      isActive 
                        ? "bg-blue-800" 
                        : "hover:bg-blue-800"
                    )}>
                      <item.icon className="w-5 h-5 mr-3" />
                      <span>{item.label}</span>
                    </div>
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
        
        <div className="mt-auto pt-8">
          <div className="bg-blue-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">Upgrade to Pro</h4>
            <p className="text-sm mb-3">Get unlimited video analysis and advanced feedback.</p>
            <Button className="w-full bg-primary hover:bg-primary/90 text-white">
              Upgrade Now
            </Button>
          </div>
        </div>
      </div>
    </aside>
  );
}
