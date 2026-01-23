import { useMode } from "@/context/ModeContext";
import ResearcherWorkspace from "@/components/workspaces/ResearcherWorkspace";
import EducatorWorkspace from "@/components/workspaces/EducatorWorkspace";
import PublicWorkspace from "@/components/workspaces/PublicWorkspace";

export default function Lab() {
  const { mode } = useMode();

  switch (mode) {
    case "educator":
      return <EducatorWorkspace />;
    case "public":
      return <PublicWorkspace />;
    case "researcher":
    default:
      return <ResearcherWorkspace />;
  }
}
