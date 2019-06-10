package simpleGA;

public class GA {

    public static void main(String[] args) {

        // Set a candidate solution
        FitnessCalc.setSolution("1111000000000000000000000000000000000000000000000000000000001111");

        // Create an initial population
        Population myPop = new Population(30, true);

        // Evolve our population until we reach an optimum solution
        int generationCount = 0;
        while (myPop.getFittest().getFitness() < .99) {
            generationCount++;
            System.out.println("Generation " + generationCount + ", average fitness: "
                    + (int) (myPop.getAverageFitness() * 100) + "%");
            myPop = Algorithm.evolvePopulation(myPop);
        }
        System.out.println("Solution found in generation " + generationCount);
        System.out.println("Genes:" + myPop.getFittest());
    }
}