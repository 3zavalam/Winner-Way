import { Trophy, BarChart2, PiggyBank } from "lucide-react";

const caseStudies = [
  {
    icon: BarChart2,
    stat: "35%",
    title: "Better Rally Consistency",
    description: "Beta users improved their rally consistency by 35% after practicing with AI-guided drills."
  },
  {
    icon: PiggyBank,
    stat: "$500+",
    title: "Annual Savings",
    description: "Players saved over $500 yearly by replacing private lessons with AI-driven training."
  },
  {
    icon: Trophy,
    stat: "30%",
    title: "Faster Skill Improvement",
    description: "Athletes reported a 30% quicker improvement across key skills like consistency and shot accuracy."
  }
];

const CaseStudy = () => {
  return (
    <section id="case-study" className="py-24 bg-winnerBeige">
      <div className="max-w-6xl mx-auto px-4 text-center">
        {/* Título principal */}
        <h2 className="text-3xl md:text-4xl font-bold text-winnerGreen text-center mb-12">
          Real Results from Early Players
        </h2>
        {/* Grid de estadísticas */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
          {caseStudies.map((study, index) => (
            <div key={index} className="flex flex-col items-center">
              {/* Ícono */}
              <div className="bg-winnerGreen/10 p-4 rounded-full mb-6">
                <study.icon className="h-10 w-10 text-winnerGreen" />
              </div>
              {/* Stat grande */}
              <div className="text-5xl font-extrabold text-winnerGreen mb-2">
                {study.stat}
              </div>
              {/* Título pequeño */}
              <h3 className="text-2xl font-bold text-winnerGreen mb-2">
                {study.title}
              </h3>
              {/* Descripción */}
              <p className="text-winnerGreen/80 max-w-xs">
                {study.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default CaseStudy;
