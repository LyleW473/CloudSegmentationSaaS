from src.backend.coordinator.coordinator import PipelineCoordinator

if __name__ == "__main__":

    # Queries
    test_locations = {
                    "Statue of Liberty": (40.6892, -74.0445), 
                    "River Thames": (51.5072, 0.1276),
                    "Invalid location": (200, -74.0445)
                    }
    time_of_interest = "2021-01-01/2024-01-02"

    # Coordinator
    pipeline_coordinator = PipelineCoordinator()

    for location, (latitude, longitude) in test_locations.items():
        # Format query
        query = {"site_latitude": latitude, "site_longitude": longitude, "time_of_interest": time_of_interest}

        # Run the pipeline
        result = pipeline_coordinator(query)

        print(f"Result for location: {location}:")
        if result["data"] is not None:
            print("Data retrieved successfully.")
            print("Data shape:", result["data"].shape)
        else:
            print("No data found.")