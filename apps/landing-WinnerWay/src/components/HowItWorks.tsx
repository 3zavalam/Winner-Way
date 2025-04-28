
import { CheckCircle2 } from "lucide-react";

const steps = [
  {
    title: "Record Your Swing",
    description: "Use your smartphone to capture video of your tennis strokes from various angles."
  },
  {
    title: "AI Analysis",
    description: "Our advanced AI compares your technique to thousands of professional swings."
  },
  {
    title: "Get Feedback",
    description: "Receive detailed feedback with suggestions to improve your form and technique."
  },
  {
    title: "Track Progress",
    description: "Monitor your improvement over time with historical comparisons."
  }
];

const HowItWorks = () => {
  return (
    <section id="how-it-works" className="py-16 px-4">
      <div className="max-w-5xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-winnerGreen text-center mb-12">
          How Winner Way Works
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {steps.map((step, index) => (
            <div 
              key={step.title}
              className="bg-white p-6 rounded-lg shadow-sm flex flex-col items-center text-center"
            >
              <div className="w-12 h-12 rounded-full bg-winnerGreen text-winnerBeige flex items-center justify-center mb-4 text-xl font-bold">
                {index + 1}
              </div>
              <h3 className="text-xl font-bold text-winnerGreen mb-3">{step.title}</h3>
              <p className="text-winnerGreen/80">{step.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
