from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple

# Definition of types used in the genetic algorithm
Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]

# Function to generate a random genome of a given length
def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)

# Function to generate a population of random genomes
def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

# Function for single-point crossover between two genomes
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of the same length")

    length = len(a)
    if length < 2:
        return a, b

    # Randomly choose a crossover point
    p = randint(1, length - 1)
    # Swap the genetic material between parents at the crossover point
    return a[0:p] + b[p:], b[0:p] + a[p:]

# Function to perform mutation on a genome
def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        # Randomly choose a gene index to mutate
        index = randrange(len(genome))
        # Flip the selected gene with a certain probability
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome

# Function to calculate the total fitness of a population
def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])

# Function for selecting a pair of individuals from the population based on their fitness
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(gene) for gene in population],
        k=2
    )

# Function to sort the population based on fitness in descending order
def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)

# Function to convert a genome to a string
def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))

# Function to print statistics about the current population
def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]),
                              fitness_func(sorted_population[-1])))
    print("")

    return sorted_population[0]

# Function to run the genetic algorithm
def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    # Initialize the population
    population = populate_func()

    # Iterate through generations
    for i in range(generation_limit):
        # Sort the population based on fitness in descending order
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        # Print statistics if a printer function is provided
        if printer is not None:
            printer(population, i, fitness_func)

        # Check if the best individual meets the fitness limit
        if fitness_func(population[0]) >= fitness_limit:
            break

        # Select the top 2 individuals from the current population
        next_generation = population[0:2]

        # Generate offspring through selection, crossover, and mutation
        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        # Update the population for the next generation
        population = next_generation

    # Return the final population and the generation count
    return population, i
# Define a simple fitness function (count the number of ones in the genome)
def simple_fitness(genome: Genome) -> int:
    return sum(genome)

# Define a printer function (optional, you can also use the default print_stats)
def custom_printer(population: Population, generation_id: int, fitness_func: FitnessFunc):
    best_genome = print_stats(population, generation_id, fitness_func)
    print("Custom Additional Info: %s" % (best_genome))

# Test the genetic algorithm
population, generations = run_evolution(
    populate_func=lambda: generate_population(size=10, genome_length=8),
    fitness_func=simple_fitness,
    fitness_limit=8,  # Adjust as needed based on your fitness function
    printer=custom_printer
)

# Print final results
print("Final Population: %s" % (population))
print("Number of Generations: %d" % (generations))
