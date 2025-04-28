
import { CheckCircle } from "lucide-react";

const benefits = [
  {
    title: "Pro-Level Feedback",
    description: "Get technique analysis comparable to working with a professional coach."
  },
  {
    title: "Train Anytime",
    description: "Practice and receive feedback whenever and wherever you play."
  },
  {
    title: "Track Progress",
    description: "Monitor improvements in your technique over time with historical data."
  },
  {
    title: "Community Support",
    description: "Connect with other tennis enthusiasts working to improve their game."
  },
  {
    title: "Cost Effective",
    description: "Gain insights that would cost hundreds from private coaching sessions."
  },
  {
    title: "Data-Driven",
    description: "Make improvements based on objective analysis, not just feel."
  }
];

const Benefits = () => {
  return (
    <section id="benefits" className="py-16 px-4">
      <div className="max-w-5xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-winnerGreen text-center mb-12">
          Benefits of Winner Way
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {benefits.map((benefit) => (
            <div key={benefit.title} className="flex space-x-4">
              <CheckCircle className="h-6 w-6 text-winnerGreen flex-shrink-0 mt-1" />
              <div>
                <h3 className="text-xl font-bold text-winnerGreen mb-2">{benefit.title}</h3>
                <p className="text-winnerGreen/80">{benefit.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Benefits;
