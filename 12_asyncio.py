import asyncio


async def main():
    print("hello")
    await asyncio.sleep(2)
    print("world")


async def main1():
    print("namaste")
    await asyncio.sleep(2)
    print("duniya")

    name_tool = {}
    name_tool["name"] = "John Doe"
    name_tool["age"] = 30
    name_tool["city"] = "New York"
    print(name_tool["name"])


async def run():
    await asyncio.gather(main1(), main())

if __name__ == "__main__":
    asyncio.run(run())
